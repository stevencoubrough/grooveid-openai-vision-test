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
    You can pass a public image_url or a base64 encoded image (JPEG).
    """
    if not image_url and not image_b64:
        raise ValueError("Must supply image_url or image_b64")

    # Build the image part for chat completions
    if image_b64:
        data_uri = f"data:image/jpeg;base64,{image_b64}"
        img_part = {
            "type": "image_url",
            "image_url": {"url": data_uri}
        }
    else:
        img_part = {
            "type": "image_url",
            "image_url": {"url": image_url}
        }

    messages = [
        {"role": "system", "content": VISION_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": "Extract metadata as per schema."},
            img_part
        ]}
    ]

    # Call Chat Completions API to get JSON result
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=600
    )

    # Extract the JSON text
    text = resp.choices[0].message.content

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        cleaned = text.strip().removeprefix("```json").removesuffix("```").strip()
        data = json.loads(cleaned)

    return VisionResult(data=data)
