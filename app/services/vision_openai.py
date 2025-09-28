# app/services/vision_openai.py

from typing import Optional, Dict, Any
import os
import json

from openai import OpenAI

MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
_client = OpenAI()

VISION_SYSTEM_PROMPT = """
You extract record metadata from label images and return strict JSON only
matching this schema:

{
  "artist": string|null,
  "title": string|null,
  "label": string|null,
  "catalogNo": string|null,
  "year": string|null,
  "country": string|null,
  "keywords": string[],
  "raw_text": string|null
}

Rules:
- Prefer exact text on the label for catalogNo and label.
- If multiple candidates exist, choose the most prominent and include alternates in keywords.
- No commentary and no keys outside the schema.
- Output MUST be valid JSON without code fences.
"""

def _to_data_uri(image_b64: str) -> str:
    return f"data:image/jpeg;base64,{image_b64}"

async def extract_from_image(
    image_url: Optional[str] = None,
    image_b64: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract structured record metadata from an image via Chat Completions.
    Accepts either a public image URL or a base64-encoded JPEG (no header).
    Returns a Python dict per the schema in VISION_SYSTEM_PROMPT.
    """
    if not image_url and not image_b64:
        raise ValueError("extract_from_image: provide image_url or image_b64")

    if image_b64:
        image_part = {"type": "image_url", "image_url": {"url": _to_data_uri(image_b64)}}
    else:
        image_part = {"type": "image_url", "image_url": {"url": image_url}}

    messages = [
        {"role": "system", "content": VISION_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": "Extract metadata as per schema."},
            image_part
        ]}
    ]

    resp = _client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=600,
    )

    text = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        cleaned = text.strip("` \n").replace("json\n", "").replace("json\r", "")
        data = json.loads(cleaned)

    for k in ("artist", "title", "label", "catalogNo", "year", "country", "keywords", "raw_text"):
        data.setdefault(k, None if k != "keywords" else [])

    return data
