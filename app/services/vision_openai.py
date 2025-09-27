import base64
import os
import json
from typing import Optional, Dict, Any

from openai import OpenAI
from pydantic import BaseModel

MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")

client = OpenAI()

# System prompt guiding the vision extraction
VISION_SYSTEM_PROMPT = """You extract record metadata from label images.
Return strict JSON only, matching this schema:
{
  "artist": string|null,
  "title": string|null,
  "label": string|null,
  "catalogNo": string|null,
  "year": string|null,
  "country": string|null,
  "keywords": string[],   // searchable terms: artist, title, label, cat no variants
  "raw_text": string|null,
  "confidence_notes": string|null
}
Rules:
- Prefer exact text on the label for catalogNo and label.
- If multiple candidates exist, choose the most prominent and include alternates in keywords.
- Keep JSON minimal, no additional keys, no commentary outside JSON.
"""

class VisionResult(BaseModel):
    data: Dict[str, Any]

async def extract_from_image(image_url: Optional[str] = None, image_b64: Optional[str] = None) -> VisionResult:
    """
    Uses OpenAI Vision to extract structured metadata from a record label image.
    You can pass a public image_url OR a base64 encoded image (JPEG).
    """
    if not image_url and not image_b64:
        raise ValueError("Must supply image_url or image_b64")

    if image_b64:
        img_part = {
            "type": "input_image",
            "image": {"data": image_b64, "mime_type": "image/jpeg"}
        }
    else:
        img_part = {
            "type": "input_image",
            "image_url": image_url
        }

    # Send a multimodal request
    resp = client.responses.create(
        model=MODEL,
        input=[{
            "role": "system",
            "content": VISION_SYSTEM_PROMPT
        },{
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract metadata as per schema."},
                img_part
            ]
        }],
        temperature=0.0,
        max_output_tokens=600
    )

    # unified text accessor in v1 SDK
    text = resp.output_text

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # fallback: strip fences
        cleaned = text.strip().removeprefix("```json").removesuffix("```").strip()
        data = json.loads(cleaned)

    return VisionResult(data=data)
