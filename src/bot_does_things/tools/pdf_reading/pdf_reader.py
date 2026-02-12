import json
import re
import numpy as np
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pdfplumber
from tqdm import tqdm
from pdfplumber.utils import extract_text

from bot_does_things.tools.pdf_reading.surya_images import (
    extract_figures_and_exclusion_bboxes,
)


BBox = tuple[float, float, float, float]


THRESHOLDS = {
    "maybe_title": {
        "min_len": 4,
        "max_len": 140,
        "max_caps_words": 12,
        "caps_ratio": 0.85,
    },
    "underline": {
        "y_tolerance": 3.0,
        "min_overlap_ratio": 0.7,
    },
    "line_grouping": {
        "top_tolerance": 2.0,
    },
    "title": {
        "max_width_to_median": 0.92,
    },
    "table_legend": {
        "lines_above": 5,
    },
    "header_footer": {
        "top_region_frac": 0.12,
        "bottom_region_frac": 0.12,
        "min_page_fraction": 0.4,
    },
    "figures": {
        "dpi": 96,
    },
}


TABLE_LEGEND_RE = re.compile(
    r"^Table\s+[0-9.]+(?:(?:\.|:)|(?:\s+-\s+)|(?:\s+[—–]\s+)|(?:\s+[—–]))",
    flags=re.IGNORECASE,
)

FIGURE_LEGEND_RE = re.compile(
    r"^Figure\s+[0-9.]+(?:(?:\.|:)|(?:\s+-\s+)|(?:\s+[—–]\s+)|(?:\s+[—–]))",
    flags=re.IGNORECASE,
)


def _pages_with_images(elements: list[dict[str, Any]]) -> list[int]:
    pages: set[int] = set()
    for el in elements:
        page = el.get("page_number")
        if not isinstance(page, int):
            continue
        text = (el.get("text") or "").strip()
        if not text:
            continue
        if FIGURE_LEGEND_RE.match(text):
            pages.add(page)
    return sorted(pages)


def _normalize_heading(text: str) -> str:
    return " ".join(text.strip().split())


def _maybe_title_line(line: str) -> bool:
    line = line.strip()
    if not line:
        return False

    if len(line) < THRESHOLDS["maybe_title"]["min_len"]:
        return False
    if len(line) > THRESHOLDS["maybe_title"]["max_len"]:
        return False

    numbered = re.match(r"^(\d+(?:\.\d+)*)\s+\S+", line)
    if numbered:
        return True

    words = re.findall(r"[A-Za-z]+", line)
    if not words:
        return False
    upper_ratio = sum(1 for c in line if c.isalpha() and c.isupper()) / max(
        1, sum(1 for c in line if c.isalpha())
    )
    if (
        upper_ratio > THRESHOLDS["maybe_title"]["caps_ratio"]
        and len(words) <= THRESHOLDS["maybe_title"]["max_caps_words"]
    ):
        return True

    return False


def _line_height_stats(elements: list[dict[str, Any]]) -> dict[str, Any]:
    heights: list[float] = []
    for el in elements:
        md = el.get("metadata")
        if not isinstance(md, dict):
            continue
        style = md.get("style")
        if not isinstance(style, dict):
            continue
        h = style.get("line_height")
        if h is None:
            continue
        try:
            hf = float(h)
        except (TypeError, ValueError):
            continue
        if hf <= 0:
            continue
        heights.append(hf)

    return {
        "count": len(heights),
        "mean": float(np.mean(heights)),
        "median": float(np.median(heights)),
    }


def _normalize_repeated_text(text: str) -> str:
    text = " ".join(text.strip().split()).lower()
    text = re.sub(r"\d+", "<num>", text)
    return text


def _classify_headers_footers(
    elements: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not elements:
        return elements, {
            "pages": 0,
            "header": None,
            "footer": None,
        }

    # Pick at most one candidate header/footer per page based on position.
    per_page: dict[int, dict[str, Any]] = {}
    for idx, el in enumerate(elements):
        page = el.get("page_number")
        if not isinstance(page, int):
            continue
        text = (el.get("text") or "").strip()
        if not text:
            continue
        md = el.get("metadata")
        if not isinstance(md, dict):
            continue
        style = md.get("style")
        if not isinstance(style, dict):
            continue
        top = style.get("top")
        bottom = style.get("bottom")
        page_height = style.get("page_height")
        if top is None or bottom is None or page_height is None:
            continue
        try:
            top_f = float(top)
            bottom_f = float(bottom)
            page_h = float(page_height)
        except (TypeError, ValueError):
            continue
        if page_h <= 0:
            continue

        entry = per_page.setdefault(
            page,
            {
                "header": None,
                "footer": None,
            },
        )

        top_limit = page_h * float(
            THRESHOLDS["header_footer"]["top_region_frac"]
        )
        bottom_limit = page_h * float(
            THRESHOLDS["header_footer"]["bottom_region_frac"]
        )

        if top_f <= top_limit:
            existing = entry["header"]
            if existing is None or top_f < existing["top"]:
                entry["header"] = {
                    "idx": idx,
                    "top": top_f,
                    "text": text,
                    "norm": _normalize_repeated_text(text),
                }

        if bottom_f >= page_h - bottom_limit:
            existing = entry["footer"]
            # Choose the bottom-most line as footer candidate.
            if existing is None or bottom_f > existing["bottom"]:
                entry["footer"] = {
                    "idx": idx,
                    "bottom": bottom_f,
                    "text": text,
                    "norm": _normalize_repeated_text(text),
                }

    pages = len(per_page)
    if pages == 0:
        return elements, {
            "pages": 0,
            "header": None,
            "footer": None,
        }

    header_counts: Counter[str] = Counter()
    footer_counts: Counter[str] = Counter()
    for entry in per_page.values():
        if entry.get("header"):
            header_counts[entry["header"]["norm"]] += 1
        if entry.get("footer"):
            footer_counts[entry["footer"]["norm"]] += 1

    min_fraction = float(THRESHOLDS["header_footer"]["min_page_fraction"])
    min_count = int(np.ceil(pages * min_fraction))

    def pick_pattern(counts: Counter[str]) -> tuple[str | None, int]:
        if not counts:
            return None, 0
        pattern, cnt = counts.most_common(1)[0]
        if cnt < min_count:
            return None, cnt
        return pattern, cnt

    def pick_patterns(
        counts: Counter[str], max_patterns: int
    ) -> list[dict[str, Any]]:
        picked: list[dict[str, Any]] = []
        for pattern, cnt in counts.most_common(max_patterns):
            if cnt < min_count:
                continue
            picked.append(
                {
                    "pattern": pattern,
                    "count": cnt,
                    "fraction": cnt / pages if pages else None,
                }
            )
        return picked

    header_pattern, header_count = pick_pattern(header_counts)
    footer_patterns = pick_patterns(footer_counts, max_patterns=2)
    footer_pattern_set = {p["pattern"] for p in footer_patterns}

    # Apply labels: at most one per page (candidate chosen above).
    for page, entry in per_page.items():
        if header_pattern and entry.get("header"):
            if entry["header"]["norm"] == header_pattern:
                elements[entry["header"]["idx"]]["type"] = "Header"
        if footer_patterns and entry.get("footer"):
            if entry["footer"]["norm"] in footer_pattern_set:
                elements[entry["footer"]["idx"]]["type"] = "Footer"

    stats = {
        "pages": pages,
        "min_page_fraction": min_fraction,
        "header": (
            {
                "pattern": header_pattern,
                "count": header_count,
                "fraction": header_count / pages if pages else None,
            }
            if header_pattern
            else None
        ),
        "footer": footer_patterns if footer_patterns else None,
    }
    return elements, stats


def _merge_tables_across_pages(
    elements: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not elements:
        return elements, {"merged_tables": 0}

    deleted: set[int] = set()
    merged_tables = 0

    def next_non_hf_index(start: int) -> int | None:
        j = start
        while j < len(elements):
            t = elements[j].get("type")
            if t in {"Header", "Footer"}:
                j += 1
                continue
            return j
        return None

    i = 0
    while i < len(elements):
        if i in deleted:
            i += 1
            continue
        el = elements[i]
        if el.get("type") != "Table":
            i += 1
            continue

        group = [i]
        last_table_index = i
        while True:
            j = next_non_hf_index(last_table_index + 1)
            if j is None or j in deleted:
                break
            nxt = elements[j]
            if nxt.get("type") != "Table":
                break
            if nxt.get("page_number") == el.get("page_number"):
                break

            group.append(j)
            last_table_index = j

        if len(group) <= 1:
            i += 1
            continue

        base = elements[group[0]]
        base_md = base.get("metadata")
        if not isinstance(base_md, dict):
            base_md = {}
            base["metadata"] = base_md

        pages: list[int] = []
        bboxes: list[Any] = []
        merged_data: list[Any] = []

        for gi in group:
            tbl = elements[gi]
            md = tbl.get("metadata")
            if isinstance(md, dict):
                p = md.get("page_number")
                if isinstance(p, int):
                    pages.append(p)
                bb = md.get("bbox")
                if bb is not None:
                    bboxes.append(bb)
                data = md.get("table")
                if isinstance(data, list):
                    merged_data.extend(data)
            else:
                p = tbl.get("page_number")
                if isinstance(p, int):
                    pages.append(p)

        if merged_data:
            base_md["table"] = merged_data
        if pages:
            base_md["pages"] = sorted(set(pages))
        if bboxes:
            base_md["bboxes"] = bboxes

        for gi in group[1:]:
            deleted.add(gi)
        merged_tables += 1

        i = group[-1] + 1

    if not deleted:
        for new_idx, el in enumerate(elements):
            el["index"] = new_idx
        return elements, {"merged_tables": 0}

    new_elements = [el for idx, el in enumerate(elements) if idx not in deleted]
    for new_idx, el in enumerate(new_elements):
        el["index"] = new_idx
    return new_elements, {"merged_tables": merged_tables}


def _char_in_bbox(char: dict[str, Any], bbox: BBox) -> bool:
    x0, top, x1, bottom = bbox
    cx0 = char.get("x0")
    cx1 = char.get("x1")
    ctop = char.get("top")
    cbottom = char.get("bottom")
    if cx0 is None or cx1 is None or ctop is None or cbottom is None:
        return False
    return cx0 >= x0 and cx1 <= x1 and ctop >= top and cbottom <= bottom


def _is_bold_font(fontname: str | None) -> bool:
    if not fontname:
        return False
    return "bold" in fontname.lower()


def _is_italic_font(fontname: str | None) -> bool:
    if not fontname:
        return False
    name = fontname.lower()
    return "italic" in name or "oblique" in name


def _is_underlined(
    page: Any,
    x0: float,
    x1: float,
    bottom: float,
) -> bool:
    lines = getattr(page, "lines", None)
    if not lines:
        return False
    # pdfplumber uses top-origin coordinates for lines too.
    # Consider an underline if a horizontal line is just below the text.
    for ln in lines:
        if ln.get("orientation") != "h":
            continue
        y = ln.get("top")
        if y is None:
            continue
        if (
            abs(float(y) - float(bottom))
            > THRESHOLDS["underline"]["y_tolerance"]
        ):
            continue
        lx0 = ln.get("x0")
        lx1 = ln.get("x1")
        if lx0 is None or lx1 is None:
            continue
        overlap = min(float(lx1), x1) - max(float(lx0), x0)
        if overlap <= 0:
            continue
        if (
            overlap / max(1.0, x1 - x0)
            >= THRESHOLDS["underline"]["min_overlap_ratio"]
        ):
            return True
    return False


def _classify_titles_and_merge(
    elements: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    text_like = [
        el
        for el in elements
        if el.get("type") == "Text"
        and el.get("text")
        and el.get("metadata", {}).get("_style")
    ]
    if not text_like:
        for i, el in enumerate(elements):
            el["index"] = i
        return elements

    sizes = [
        float(el["metadata"]["_style"].get("size", 0.0)) for el in text_like
    ]
    # Estimate body font size as the MODE (rounded to reduce floating noise).
    rounded_sizes = [round(s, 1) for s in sizes if s > 0]
    size_counts = Counter(rounded_sizes)
    body_size = size_counts.most_common(1)[0][0] if size_counts else 0.0

    # Estimate body line width as the MEDIAN line width among lines that look like body text.
    body_width_candidates: list[float] = []
    for el in text_like:
        style = el.get("metadata", {}).get("_style", {})
        size = float(style.get("size", 0.0))
        width = float(style.get("line_width", 0.0) or 0.0)
        if width <= 0:
            continue
        # Only consider lines that match the body font size and have no emphasis.
        if round(size, 1) != body_size:
            continue
        if style.get("bold") or style.get("italic") or style.get("underline"):
            continue
        body_width_candidates.append(width)

    body_width_candidates.sort()
    if body_width_candidates:
        median_body_width = body_width_candidates[
            len(body_width_candidates) // 2
        ]
    else:
        # Fallback: median across all line widths.
        all_widths = [
            float(
                el.get("metadata", {}).get("_style", {}).get("line_width", 0.0)
                or 0.0
            )
            for el in text_like
        ]
        all_widths = [w for w in all_widths if w > 0]
        all_widths.sort()
        median_body_width = (
            all_widths[len(all_widths) // 2] if all_widths else 1.0
        )

    # Fill width_to_median for all text lines.
    for el in text_like:
        style = el.get("metadata", {}).get("_style", {})
        width = float(style.get("line_width", 0.0) or 0.0)
        style["width_to_median"] = width / max(1.0, float(median_body_width))

    def compute_title_fields(el: dict[str, Any]) -> None:
        style = el.get("metadata", {}).get("_style", {})
        size = float(style.get("size", 0.0))
        text = (el.get("text") or "").strip()
        width_to_median = float(style.get("width_to_median", 1.0))

        has_emphasis = bool(
            style.get("bold") or style.get("italic") or style.get("underline")
        )

        is_title = False
        # Heuristics:
        # - Titles usually do not end with '.'
        # - Titles usually do not span full body text width (relative to median body width)
        # - Font size should be larger than body size, OR be emphasized and look like a heading.
        if (
            text
            and not text.endswith(".")
            and width_to_median < THRESHOLDS["title"]["max_width_to_median"]
        ):
            if round(size, 1) > float(body_size):
                is_title = True
            elif (
                round(size, 1) == float(body_size)
                and has_emphasis
                and _maybe_title_line(text)
            ):
                is_title = True

        el["metadata"]["_is_title_candidate"] = is_title
        el["metadata"]["_title_score"] = {
            "size": size,
            "bold": bool(style.get("bold")),
            "italic": bool(style.get("italic")),
            "underline": bool(style.get("underline")),
            "no_period": bool(text and not text.endswith(".")),
            "body_size": float(body_size),
            "width_to_median": width_to_median,
            "narrow_bonus": max(0.0, 1.0 - width_to_median),
        }

    # Compute title candidate + scoring quantities for every extracted line.
    for el in text_like:
        compute_title_fields(el)

    merged: list[dict[str, Any]] = []
    i = 0
    while i < len(elements):
        el = elements[i]
        if el.get("type") != "Text" or not el.get("metadata", {}).get("_style"):
            merged.append(el)
            i += 1
            continue

        style = el["metadata"]["_style"]
        if not el["metadata"].get("_is_title_candidate"):
            merged.append(el)
            i += 1
            continue

        # Promote to Title and merge subsequent identically styled lines.
        title_lines = [(el.get("text") or "").strip()]
        el["type"] = "Title"

        j = i + 1
        while j < len(elements):
            nxt = elements[j]
            if nxt.get("type") != "Text" or not nxt.get("metadata", {}).get(
                "_style"
            ):
                break
            nxt_style = nxt["metadata"]["_style"]
            if nxt_style != style:
                break
            t = (nxt.get("text") or "").strip()
            if not t:
                j += 1
                continue
            title_lines.append(t)
            j += 1

        el["text"] = "\n".join([t for t in title_lines if t])
        merged.append(el)
        i = j

    # Export quantities (style + scores) and reindex.
    for new_idx, el in enumerate(merged):
        md = el.get("metadata")
        if isinstance(md, dict) and "_style" in md:
            md["style"] = md.get("_style")
            md["title_candidate"] = bool(md.get("_is_title_candidate", False))
            md["title_score"] = md.get("_title_score")
            md.pop("_style", None)
            md.pop("_is_title_candidate", None)
            md.pop("_title_score", None)
        el["index"] = new_idx
    return merged


def _classify_table_legends(
    elements: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not elements:
        return elements

    n_above = int(THRESHOLDS["table_legend"]["lines_above"])
    i = 0
    while i < len(elements):
        el = elements[i]
        if el.get("type") != "Table":
            i += 1
            continue

        table_page = el.get("page_number")
        hit_index: int | None = None

        # Scan up to N lines above the table.
        for k in range(1, n_above + 1):
            j = i - k
            if j < 0:
                break
            prev = elements[j]
            if prev.get("page_number") != table_page:
                break
            prev_type = prev.get("type")
            if prev_type != "Text":
                # Stop on Title/Table/any other non-Text element.
                break
            prev_text = (prev.get("text") or "").strip()
            if not prev_text:
                continue
            if TABLE_LEGEND_RE.match(prev_text):
                hit_index = j
                break

        if hit_index is None:
            i += 1
            continue

        # Merge from hit_index up to the line immediately above the table.
        start = hit_index
        end = i - 1
        merged_lines: list[str] = []
        base_metadata: dict[str, Any] | None = None
        ok = True
        for j in range(start, end + 1):
            cur = elements[j]
            if cur.get("page_number") != table_page:
                ok = False
                break
            if cur.get("type") != "Text":
                ok = False
                break
            t = (cur.get("text") or "").strip()
            if t:
                merged_lines.append(t)
            if base_metadata is None and isinstance(cur.get("metadata"), dict):
                base_metadata = dict(cur["metadata"])

        if not ok or not merged_lines:
            i += 1
            continue

        legend_el: dict[str, Any] = {
            "type": "TableLegend",
            "text": "\n".join(merged_lines),
            "metadata": base_metadata or {"page_number": table_page},
            "page_number": table_page,
        }

        # Replace the slice [start:end] with a single TableLegend.
        elements[start : end + 1] = [legend_el]

        # Table moved left; continue scanning after the table.
        i = start + 1

    for new_idx, el in enumerate(elements):
        el["index"] = new_idx
    return elements


def extract_elements(
    input_path: Path,
    extract_tables: bool,
    split_titles: bool,
    exclude_bboxes_by_page: dict[int, list[BBox]] | None = None,
) -> list[dict[str, Any]]:
    elements: list[dict[str, Any]] = []
    idx = 0

    def _detect_column_gutters(
        chars: list[dict[str, Any]],
        page_width: float,
    ) -> list[float]:
        if not chars or page_width <= 1.0:
            return []

        xs: list[float] = []
        for ch in chars:
            x0 = ch.get("x0")
            x1 = ch.get("x1")
            if x0 is None or x1 is None:
                continue
            xs.append(0.5 * (float(x0) + float(x1)))
        if len(xs) < 200:
            return []

        bins = 120
        hist, edges = np.histogram(
            xs, bins=bins, range=(0.0, float(page_width))
        )
        window = 7
        kernel = np.ones(window, dtype=float) / float(window)
        smooth = np.convolve(hist.astype(float), kernel, mode="same")

        positive = smooth[smooth > 0]
        if positive.size == 0:
            return []

        low_thr = float(np.percentile(positive, 10))
        low = smooth <= low_thr

        min_x = 0.08 * float(page_width)
        max_x = 0.92 * float(page_width)

        gutters: list[tuple[int, int]] = []
        start: int | None = None
        for i, is_low in enumerate(low.tolist()):
            if is_low and start is None:
                start = i
                continue
            if (not is_low) and start is not None:
                gutters.append((start, i - 1))
                start = None
        if start is not None:
            gutters.append((start, bins - 1))

        centers: list[float] = []
        for a, b in gutters:
            if (b - a + 1) < 3:
                continue
            x0 = float(edges[a])
            x1 = float(edges[b + 1])
            cx = 0.5 * (x0 + x1)
            if cx < min_x or cx > max_x:
                continue
            centers.append(cx)

        centers = sorted(centers)
        merged: list[float] = []
        for c in centers:
            if not merged:
                merged.append(c)
                continue
            if abs(c - merged[-1]) <= 0.06 * float(page_width):
                merged[-1] = 0.5 * (merged[-1] + c)
                continue
            merged.append(c)

        return merged

    def _column_ranges_from_gutters(
        gutters: list[float],
        page_width: float,
    ) -> list[tuple[float, float]]:
        if not gutters:
            return [(0.0, float(page_width))]
        cuts = [0.0] + sorted([float(g) for g in gutters]) + [float(page_width)]
        ranges: list[tuple[float, float]] = []
        for i in range(len(cuts) - 1):
            ranges.append((cuts[i], cuts[i + 1]))
        return ranges

    def _pick_column_ranges(
        chars: list[dict[str, Any]],
        page_width: float,
    ) -> list[tuple[float, float]]:
        gutters = _detect_column_gutters(chars, page_width)
        if not gutters:
            return [(0.0, float(page_width))]

        xs: list[float] = []
        for ch in chars:
            x0 = ch.get("x0")
            x1 = ch.get("x1")
            if x0 is None or x1 is None:
                continue
            xs.append(0.5 * (float(x0) + float(x1)))
        if not xs:
            return [(0.0, float(page_width))]

        def counts_for(guts: list[float]) -> list[int]:
            ranges = _column_ranges_from_gutters(guts, page_width)
            counts = [0 for _ in ranges]
            for x in xs:
                for i, (a, b) in enumerate(ranges):
                    if (i == len(ranges) - 1 and a <= x <= b) or (a <= x < b):
                        counts[i] += 1
                        break
            return counts

        total = float(len(xs))
        min_frac = 0.18

        best = [(0.0, float(page_width))]

        # Try 3 columns (2 gutters).
        if len(gutters) >= 2:
            candidate = [gutters[0], gutters[1]]
            counts = counts_for(candidate)
            if len(counts) == 3 and all(
                (c / total) >= min_frac for c in counts
            ):
                return _column_ranges_from_gutters(candidate, page_width)

        # Try 2 columns (1 gutter).
        candidate = [gutters[0]]
        counts = counts_for(candidate)
        if len(counts) == 2 and all((c / total) >= min_frac for c in counts):
            best = _column_ranges_from_gutters(candidate, page_width)

        return best

    def build_lines_from_chars(
        chars: list[dict[str, Any]],
        page_number: int,
        page: Any,
    ) -> list[dict[str, Any]]:
        if not chars:
            return []

        # Group chars into lines by their "top" coordinate.
        sorted_chars = sorted(
            chars, key=lambda c: (c.get("top", 0.0), c.get("x0", 0.0))
        )
        lines: list[dict[str, Any]] = []
        line_chars: list[dict[str, Any]] = []
        current_top: float | None = None
        tolerance = THRESHOLDS["line_grouping"]["top_tolerance"]

        def flush_line() -> None:
            nonlocal line_chars
            nonlocal current_top
            if not line_chars or current_top is None:
                line_chars = []
                current_top = None
                return

            text = extract_text(line_chars) or ""
            text = _normalize_heading(text)
            if text:
                sizes = [
                    c.get("size")
                    for c in line_chars
                    if c.get("size") is not None
                ]
                size = float(max(sizes) if sizes else 0.0)
                bold = any(_is_bold_font(c.get("fontname")) for c in line_chars)
                italic = any(
                    _is_italic_font(c.get("fontname")) for c in line_chars
                )
                x0s = [
                    c.get("x0") for c in line_chars if c.get("x0") is not None
                ]
                x1s = [
                    c.get("x1") for c in line_chars if c.get("x1") is not None
                ]
                tops = [
                    c.get("top") for c in line_chars if c.get("top") is not None
                ]
                bottoms = [
                    c.get("bottom")
                    for c in line_chars
                    if c.get("bottom") is not None
                ]
                x0 = float(min(x0s)) if x0s else 0.0
                x1 = (
                    float(max(x1s))
                    if x1s
                    else float(getattr(page, "width", 1.0))
                )
                top = float(min(tops)) if tops else float(current_top)
                bottom = float(max(bottoms)) if bottoms else top
                line_width = float(x1 - x0)
                line_height = float(bottom - top)
                underline = _is_underlined(page, x0=x0, x1=x1, bottom=bottom)
                page_height = float(getattr(page, "height", 0.0) or 0.0)

                lines.append(
                    {
                        "type": "Text",
                        "text": text,
                        "metadata": {
                            "page_number": page_number,
                            "_style": {
                                "size": size,
                                "bold": bool(bold),
                                "italic": bool(italic),
                                "underline": bool(underline),
                                "line_width": line_width,
                                "line_height": line_height,
                                "x0": float(x0),
                                "x1": float(x1),
                                "top": float(top),
                                "bottom": float(bottom),
                                "page_height": page_height,
                            },
                        },
                        "page_number": page_number,
                        "_y": float(top),
                    }
                )
            line_chars = []
            current_top = None

        for ch in sorted_chars:
            top = ch.get("top")
            if top is None:
                continue
            if current_top is None:
                current_top = float(top)
                line_chars = [ch]
                continue

            if abs(float(top) - float(current_top)) <= tolerance:
                line_chars.append(ch)
                continue

            flush_line()
            current_top = float(top)
            line_chars = [ch]

        flush_line()
        return lines

    with pdfplumber.open(str(input_path)) as pdf:
        for page_index, page in tqdm(enumerate(pdf.pages, start=1)):
            page_width = float(getattr(page, "width", 0.0) or 0.0)
            page_items: list[dict[str, Any]] = []

            # Detect tables with bboxes so we can (a) exclude their text and (b) insert the
            # Table element at the correct position in the reading order.
            table_bboxes: list[BBox] = []
            page_tables = []
            if extract_tables:
                try:
                    page_tables = page.find_tables() or []
                except Exception:
                    page_tables = []
                for t in page_tables:
                    if getattr(t, "bbox", None):
                        table_bboxes.append(t.bbox)

            filtered_chars: list[dict[str, Any]] = []
            if getattr(page, "chars", None):
                excluded_bboxes = (exclude_bboxes_by_page or {}).get(
                    page_index, []
                )
                for ch in page.chars:
                    if table_bboxes and any(
                        _char_in_bbox(ch, bbox) for bbox in table_bboxes
                    ):
                        continue
                    if excluded_bboxes and any(
                        _char_in_bbox(ch, bbox) for bbox in excluded_bboxes
                    ):
                        continue
                    filtered_chars.append(ch)

            col_ranges = _pick_column_ranges(filtered_chars, page_width)
            col_chars: list[list[dict[str, Any]]] = [[] for _ in col_ranges]
            for ch in filtered_chars:
                x0 = ch.get("x0")
                x1 = ch.get("x1")
                if x0 is None or x1 is None:
                    continue
                cx = 0.5 * (float(x0) + float(x1))
                for i, (a, b) in enumerate(col_ranges):
                    if (i == len(col_ranges) - 1 and a <= cx <= b) or (
                        a <= cx < b
                    ):
                        col_chars[i].append(ch)
                        break

            col_tables: list[list[dict[str, Any]]] = [[] for _ in col_ranges]
            if extract_tables:
                for t in page_tables:
                    bbox = getattr(t, "bbox", None)
                    if not bbox:
                        continue
                    cx = 0.5 * (float(bbox[0]) + float(bbox[2]))
                    placed = False
                    for i, (a, b) in enumerate(col_ranges):
                        if (i == len(col_ranges) - 1 and a <= cx <= b) or (
                            a <= cx < b
                        ):
                            col_tables[i].append(t)
                            placed = True
                            break
                    if not placed:
                        col_tables[0].append(t)

            for col_idx, _ in enumerate(col_ranges):
                col_items: list[dict[str, Any]] = []

                if split_titles:
                    col_items.extend(
                        build_lines_from_chars(
                            col_chars[col_idx], page_index, page
                        )
                    )
                else:
                    page_text = extract_text(col_chars[col_idx]) or ""
                    clean_text = "\n".join(
                        [l for l in page_text.splitlines() if l.strip()]
                    ).strip()
                    if clean_text:
                        col_items.append(
                            {
                                "type": "Text",
                                "text": clean_text,
                                "metadata": {"page_number": page_index},
                                "page_number": page_index,
                                "_y": 0.0,
                            }
                        )

                if extract_tables:
                    for t in col_tables[col_idx]:
                        bbox = getattr(t, "bbox", None)
                        if not bbox:
                            continue
                        try:
                            table_data = t.extract()
                        except Exception:
                            table_data = None

                        col_items.append(
                            {
                                "type": "Table",
                                "text": None,
                                "metadata": {
                                    "page_number": page_index,
                                    "table": table_data,
                                    "bbox": list(bbox),
                                },
                                "page_number": page_index,
                                "_y": float(bbox[1]),
                            }
                        )

                # Stable ordering within column: sort by y; if equal, keep text before tables.
                def sort_key(item: dict[str, Any]) -> tuple[float, int]:
                    y = float(item.get("_y", 0.0))
                    type_rank = 1 if item.get("type") == "Table" else 0
                    return (y, type_rank)

                col_items.sort(key=sort_key)
                page_items.extend(col_items)
            for item in page_items:
                item.pop("_y", None)
                item["index"] = idx
                idx += 1
                elements.append(item)

    if split_titles:
        elements = _classify_titles_and_merge(elements)
        elements = _classify_table_legends(elements)
        return elements
    for i, el in enumerate(elements):
        el["index"] = i
    return elements


def extract_pdf(
    input_path: str,
    output_dir: str,
    extract_tables: bool = True,
    split_titles: bool = True,
    surya_dpi: int = 96,
    page_range: tuple[int, int] | None = None,
) -> str:
    """
    Extract elements from a PDF file. The structure of the output is as follows:

    - input: Path to the input PDF file.
    - created_at: Timestamp of when the extraction was performed.
    - extractor: Name of the extractor used.
    - pdfplumber: Configuration and statistics of the extraction.
      - extract_tables: Whether tables were extracted.
      - split_titles: Whether titles were split.
      - line_height_stats: Statistics of line heights.
      - header_footer: Statistics of header and footer detection.
      - table_merge: Statistics of table merging.
      - pages_with_images: List of pages with images.
      - figures: List of figures found in the PDF.
      - figures_dir: Directory where figures are saved.
      - surya_dpi: DPI used for Surya image extraction.
    - elements: List of extracted elements.

    Args:
        input_path (str):  Path to the input PDF file.
        output_dir (str): Directory to save the extracted elements.
        extract_tables (bool, optional): Whether to extract tables from the PDF.
            Defaults to True.
        split_titles (bool, optional): Whether to split titles from the PDF.
            Defaults to True.
        surya_dpi (int, optional): DPI for Surya image extraction.
            Defaults to 96.

    Returns:
        str: Path to the output JSON file.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "results.json"

    # First pass: extract elements (no figure exclusions) to find candidate pages.
    elements_first = extract_elements(
        input_path=input_path,
        extract_tables=extract_tables,
        split_titles=split_titles,
    )

    figures_dir = output_dir / "figures"
    figures, exclude_bboxes_by_page = extract_figures_and_exclusion_bboxes(
        pdf_path=input_path,
        elements=elements_first,
        figures_dir=figures_dir,
        dpi=int(surya_dpi),
        pages_with_images=set(_pages_with_images(elements_first)),
        figure_legend_re=FIGURE_LEGEND_RE,
        page_range=page_range,
    )

    # Second pass: exclude text inside detected figure/caption regions.
    elements = extract_elements(
        input_path=input_path,
        extract_tables=extract_tables,
        split_titles=split_titles,
        exclude_bboxes_by_page=exclude_bboxes_by_page,
    )

    line_height_stats = _line_height_stats(elements)
    elements, header_footer_stats = _classify_headers_footers(elements)
    elements, table_merge_stats = _merge_tables_across_pages(elements)
    pages_with_images = _pages_with_images(elements_first)

    payload = {
        "input": str(input_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "extractor": "pdfplumber",
        "pdfplumber": {
            "extract_tables": extract_tables,
            "split_titles": split_titles,
            "line_height_stats": line_height_stats,
            "header_footer": header_footer_stats,
            "table_merge": table_merge_stats,
            "pages_with_images": pages_with_images,
            "figures": figures,
            "figures_dir": str(figures_dir),
            "surya_dpi": int(surya_dpi),
        },
        "elements": elements,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return str(output_path)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the input PDF")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory to store results.json and extracted figures/",
    )
    parser.add_argument("--extract-tables", action="store_true", default=False)
    parser.add_argument(
        "--split-titles",
        action="store_true",
        default=True,
        help="Split text into lines and mark heading-like lines as Title",
    )
    parser.add_argument(
        "--no-split-titles", action="store_false", dest="split_titles"
    )
    parser.add_argument(
        "--surya-dpi",
        type=int,
        default=int(THRESHOLDS["figures"]["dpi"]),
        help="DPI used to rasterize PDF pages for Surya layout",
    )
    parser.add_argument(
        "--page_range",
        type=int,
        default=None,
        help="Range of pages to process",
        nargs=2,
    )
    args = parser.parse_args()

    extract_pdf(
        input_path=args.input,
        output_dir=args.output_dir,
        extract_tables=args.extract_tables,
        split_titles=args.split_titles,
        surya_dpi=args.surya_dpi,
    )


if __name__ == "__main__":
    main()
