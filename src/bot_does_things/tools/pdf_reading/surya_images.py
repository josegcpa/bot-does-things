"""
Uses Surya to extract figures from PDFs.
"""

from pathlib import Path
from typing import Any

import pdfplumber
from PIL import Image

from surya.foundation import FoundationPredictor
from surya.debug.draw import draw_polys_on_image
from surya.layout import LayoutPredictor
from surya.settings import settings


BBox = tuple[float, float, float, float]


def _merge_caption_lines(
    elements: list[dict[str, Any]],
    start_idx: int,
    figure_legend_re: Any,
) -> tuple[str, list[int]]:
    """Returns merged caption text and the indices used.

    This intentionally merges the legend line and any immediately following Text
    lines on the same page.
    """
    start_el = elements[start_idx]
    page = start_el.get("page_number")
    used: list[int] = []
    parts: list[str] = []

    i = start_idx
    while i < len(elements):
        el = elements[i]
        if el.get("page_number") != page:
            break
        if el.get("type") != "Text":
            break
        text = (el.get("text") or "").strip()
        if not text:
            break
        if i != start_idx and figure_legend_re.match(text):
            break
        parts.append(text)
        used.append(i)
        i += 1

    return "\n".join(parts), used


def _pdf_bbox_to_image_bbox(
    pdf_bbox: BBox,
    page_width: float,
    page_height: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x0, top, x1, bottom = pdf_bbox
    sx = image_width / max(1.0, float(page_width))
    sy = image_height / max(1.0, float(page_height))
    ix0 = int(max(0, min(image_width, round(x0 * sx))))
    ix1 = int(max(0, min(image_width, round(x1 * sx))))
    iy0 = int(max(0, min(image_height, round(top * sy))))
    iy1 = int(max(0, min(image_height, round(bottom * sy))))
    if ix1 < ix0:
        ix0, ix1 = ix1, ix0
    if iy1 < iy0:
        iy0, iy1 = iy1, iy0
    return (ix0, iy0, ix1, iy1)


def _image_bbox_to_pdf_bbox(
    image_bbox: tuple[float, float, float, float],
    page_width: float,
    page_height: float,
    image_width: int,
    image_height: int,
) -> BBox:
    x0, y0, x1, y1 = image_bbox
    sx = image_width / max(1.0, float(page_width))
    sy = image_height / max(1.0, float(page_height))
    return (
        float(x0) / max(1e-6, sx),
        float(y0) / max(1e-6, sy),
        float(x1) / max(1e-6, sx),
        float(y1) / max(1e-6, sy),
    )


def _union_bbox(a: BBox, b: BBox) -> BBox:
    return (
        float(min(a[0], b[0])),
        float(min(a[1], b[1])),
        float(max(a[2], b[2])),
        float(max(a[3], b[3])),
    )


def extract_figures_and_exclusion_bboxes(
    *,
    pdf_path: Path,
    elements: list[dict[str, Any]],
    figures_dir: Path,
    dpi: int,
    pages_with_images: set[int],
    figure_legend_re: Any,
) -> tuple[list[dict[str, Any]], dict[int, list[BBox]]]:
    """Extract figures as PNG and return bboxes to exclude from pdfplumber text.

    This uses Surya layout detection (layout-only) to find figure/picture regions
    and pairs them with figure legend lines detected by `figure_legend_re`.

    Args:
      pdf_path: Path to the PDF.
      elements: First-pass extracted elements.
      figures_dir: Output directory for PNG crops.
      dpi: Rasterization DPI for Surya.
      pages_with_images: Candidate page numbers to process.
      figure_legend_re: Compiled regex to detect figure legends.

    Returns:
      A tuple of:
        - figures: list of figure records (path, bbox, caption)
        - exclude_bboxes_by_page: mapping from page_number to list of pdf bboxes
          that should be excluded from text extraction.
    """
    if not pages_with_images:
        return [], {}

    figures_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = figures_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    caption_indices_by_page: dict[int, list[int]] = {}
    for idx, el in enumerate(elements):
        if el.get("type") != "Text":
            continue
        page = el.get("page_number")
        if not isinstance(page, int) or page not in pages_with_images:
            continue
        text = (el.get("text") or "").strip()
        if not text:
            continue
        if figure_legend_re.match(text):
            caption_indices_by_page.setdefault(page, []).append(idx)

    if not caption_indices_by_page:
        return [], {}

    predictor = LayoutPredictor(
        FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    )

    save_dpi = 300

    figures: list[dict[str, Any]] = []
    exclude_bboxes_by_page: dict[int, list[BBox]] = {}

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_number in sorted(caption_indices_by_page.keys()):
            page = pdf.pages[page_number - 1]
            page_w = float(getattr(page, "width", 0.0) or 0.0)
            page_h = float(getattr(page, "height", 0.0) or 0.0)

            page_img_infer = page.to_image(resolution=dpi).original
            if not isinstance(page_img_infer, Image.Image):
                continue

            page_img_save = page.to_image(resolution=save_dpi).original
            if not isinstance(page_img_save, Image.Image):
                continue

            layout_results = predictor([page_img_infer])
            if not layout_results:
                continue
            layout = layout_results[0]
            bboxes = getattr(layout, "bboxes", None)
            if not bboxes:
                continue

            debug_polys: list[list[list[float]]] = []
            debug_labels: list[str] = []
            debug_colors: list[str] = []

            sx = float(page_img_save.width) / max(
                1.0, float(page_img_infer.width)
            )
            sy = float(page_img_save.height) / max(
                1.0, float(page_img_infer.height)
            )

            for bb in bboxes:
                poly = getattr(bb, "polygon", None)
                if poly and len(poly) == 4:
                    corners = [
                        [float(poly[0][0]) * sx, float(poly[0][1]) * sy],
                        [float(poly[1][0]) * sx, float(poly[1][1]) * sy],
                        [float(poly[2][0]) * sx, float(poly[2][1]) * sy],
                        [float(poly[3][0]) * sx, float(poly[3][1]) * sy],
                    ]
                else:
                    bbox = getattr(bb, "bbox", None)
                    if not bbox or len(bbox) != 4:
                        continue
                    corners = [
                        [float(bbox[0]) * sx, float(bbox[1]) * sy],
                        [float(bbox[2]) * sx, float(bbox[1]) * sy],
                        [float(bbox[2]) * sx, float(bbox[3]) * sy],
                        [float(bbox[0]) * sx, float(bbox[3]) * sy],
                    ]

                label = str(getattr(bb, "label", ""))
                color = "yellow"
                if label in {"Figure", "Picture"}:
                    color = "green"
                elif label == "Caption":
                    color = "cyan"
                elif label == "Table":
                    color = "orange"

                debug_polys.append(corners)
                debug_labels.append(label)
                debug_colors.append(color)

            figure_regions_pdf: list[BBox] = []
            for bb in bboxes:
                label = getattr(bb, "label", None)
                if label not in {"Figure", "Picture"}:
                    continue
                bbox = getattr(bb, "bbox", None)
                if not bbox or len(bbox) != 4:
                    continue
                pdf_bbox = _image_bbox_to_pdf_bbox(
                    (
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    ),
                    page_width=page_w,
                    page_height=page_h,
                    image_width=page_img_infer.width,
                    image_height=page_img_infer.height,
                )
                figure_regions_pdf.append(pdf_bbox)

            for cap_idx in caption_indices_by_page.get(page_number, []):
                caption_text, used_indices = _merge_caption_lines(
                    elements, cap_idx, figure_legend_re
                )
                if not caption_text:
                    continue

                md = elements[cap_idx].get("metadata")
                style = md.get("style") if isinstance(md, dict) else None
                if not isinstance(style, dict):
                    continue
                x0 = style.get("x0")
                x1 = style.get("x1")
                top = style.get("top")
                bottom = style.get("bottom")
                if None in {x0, x1, top, bottom}:
                    continue
                caption_bbox: BBox = (
                    float(x0),
                    float(top),
                    float(x1),
                    float(bottom),
                )

                cap_img_bbox = _pdf_bbox_to_image_bbox(
                    caption_bbox,
                    page_width=page_w,
                    page_height=page_h,
                    image_width=page_img_save.width,
                    image_height=page_img_save.height,
                )
                debug_polys.append(
                    [
                        [float(cap_img_bbox[0]), float(cap_img_bbox[1])],
                        [float(cap_img_bbox[2]), float(cap_img_bbox[1])],
                        [float(cap_img_bbox[2]), float(cap_img_bbox[3])],
                        [float(cap_img_bbox[0]), float(cap_img_bbox[3])],
                    ]
                )
                debug_labels.append("caption_bbox")
                debug_colors.append("blue")

                def dist(fig: BBox) -> float:
                    return abs(float(fig[3]) - float(caption_bbox[1]))

                nearest_fig = sorted(figure_regions_pdf, key=dist)[0]
                union_pdf = _union_bbox(nearest_fig, caption_bbox)
                exclude_bboxes_by_page.setdefault(page_number, []).append(
                    union_pdf
                )

                union_img_bbox = _pdf_bbox_to_image_bbox(
                    union_pdf,
                    page_width=page_w,
                    page_height=page_h,
                    image_width=page_img_save.width,
                    image_height=page_img_save.height,
                )
                debug_polys.append(
                    [
                        [float(union_img_bbox[0]), float(union_img_bbox[1])],
                        [float(union_img_bbox[2]), float(union_img_bbox[1])],
                        [float(union_img_bbox[2]), float(union_img_bbox[3])],
                        [float(union_img_bbox[0]), float(union_img_bbox[3])],
                    ]
                )
                debug_labels.append("union_bbox")
                debug_colors.append("red")

                crop_box = _pdf_bbox_to_image_bbox(
                    union_pdf,
                    page_width=page_w,
                    page_height=page_h,
                    image_width=page_img_save.width,
                    image_height=page_img_save.height,
                )
                cropped = page_img_save.crop(crop_box)
                out_path = (
                    figures_dir
                    / f"page-{page_number:03d}-figure-{len(figures)+1:03d}.png"
                )
                cropped.save(out_path)

                figures.append(
                    {
                        "page_number": page_number,
                        "path": str(out_path),
                        "bbox": [
                            union_pdf[0],
                            union_pdf[1],
                            union_pdf[2],
                            union_pdf[3],
                        ],
                        "caption": caption_text,
                        "caption_element_indices": used_indices,
                    }
                )

            if debug_polys:
                debug_img = page_img_save.copy()
                debug_img = draw_polys_on_image(
                    debug_polys,
                    debug_img,
                    labels=debug_labels,
                    color=debug_colors,
                )
                debug_out = debug_dir / f"page-{page_number:03d}.png"
                debug_img.save(debug_out)

    return figures, exclude_bboxes_by_page
