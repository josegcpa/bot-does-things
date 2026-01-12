"""
Includes tools to interact with images.
"""

import base64
import mimetypes
import os

from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

from bot_does_things.assertions import assert_non_empty_str, assert_file_exists
from bot_does_things.config import OLLAMA_BASE_URL
from bot_does_things.tool_wrapper import tool_wrapper


@tool_wrapper
def interpret_image(image_path: str, query: str) -> str:
    """
    Uses an LLM to interpret an image and answer a question about it.

    Args:
        image_path (str): The path to the image file.
        query (str): The question to ask about the image.

    Returns:
        str: The answer to the question.
    """
    assert_non_empty_str(image_path, "image_path")
    assert_non_empty_str(query, "query")

    model = os.environ.get("OLLAMA_IMAGE_INTERPRETATION_MODEL", "gemma3:4b")

    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "application/octet-stream"

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    llm = ChatOpenAI(
        model=model,
        openai_api_base=OLLAMA_BASE_URL,
        openai_api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
    )

    msg = HumanMessage(
        content=[
            {"type": "text", "text": query},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
            },
        ]
    )

    result = llm.invoke([msg])
    return result.content


@tool_wrapper
def ocr_image(image_path: str) -> str:
    """
    Extract text from an image using OCR.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The extracted text from the image.
    """
    p = assert_file_exists(image_path, "image_path")

    try:
        from PIL import Image  # type: ignore
    except Exception as e:
        raise ModuleNotFoundError(
            "ocr_image requires the optional dependency 'Pillow'. Install it to enable OCR."
        ) from e

    try:
        import pytesseract  # type: ignore
    except Exception as e:
        raise ModuleNotFoundError(
            "ocr_image requires the optional dependency 'pytesseract' and the system 'tesseract' binary. "
            "Install them to enable OCR."
        ) from e

    img = Image.open(str(p))
    return pytesseract.image_to_string(img)
