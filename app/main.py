# app/main.py

from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from urllib.parse import urlparse
from dotenv import load_dotenv

# Pydantic response models
from .models.schemas import IdentifyResponse, DiscogsResult

# Service layer
from .services.vision_openai import extract_from_image
from .services.discogs_client import search_candidates, rank_candidates

load_dotenv()

app = FastAPI(title="GrooveID Identify API", version="0.2.0")

# Last raw vision payload for debug
last_vision_raw: Optional[dict] = None


@app.post("/identify", response_model=IdentifyResponse)
async def identify(
    image_url: Optional[str] = Form(None),
    image_file: Optional[UploadFile] = File(None),
) -> IdentifyResponse:
    """
    Identify a record by image URL or uploaded file. Returns structured data and ranked Discogs results.
    """
    global last_vision_raw

    # Normalize and validate the image_url; treat blank or invalid as None
    raw_url = (image_url or "").strip()
    url = None
    if raw_url:
        parsed = urlparse(raw_url)
        if parsed.scheme and parsed.netloc:
            url = raw_url

    # Require either a valid URL or an uploaded file
    if not url and not image_file:
        raise HTTPException(400, "Provide either image_url or image_file")

    # Prepare source: either pass URL straight through or read/upload as base64
    b64 = None
    src_url = None

    if url:
        src_url = url
    else:
        content = await image_file.read()
        if not content:
            raise HTTPException(400, "Empty file")
        import base64
        b64 = base64.b64encode(content).decode("utf-8")

    # 1) Vision extraction
    try:
        vision = await extract_from_image(image_url=src_url, image_b64=b64)
    except Exception as e:
        # Surface a clean upstream error
        raise HTTPException(status_code=502, detail=f"Vision provider error: {e}")

    v = vision.data
    last_vision_raw = v  # keep raw for debug

    # Pull structured values (be defensive)
    artist = (v.get("artist") or "").strip() or None
    title = (v.get("title") or "").strip() or None
    label = (v.get("label") or "").strip() or None
    catno = (v.get("catalogNo") or "").strip() or None
    year = (v.get("year") or "").strip() or None
    country = (v.get("country") or "").strip() or None
    keywords: List[str] = v.get("keywords") or []

    # 2) Discogs search & ranking
    candidates_raw = await search_candidates(artist, title, label, catno, keywords)
    ranked = rank_candidates(candidates_raw, artist, title, label, catno, year, country)

    # Build response models
    c_models: List[DiscogsResult] = []
    for it in ranked:
        try:
            c_models.append(
                DiscogsResult(
                    id=it.get("id"),
                    type=it.get("type"),
                    title=it.get("title"),
                    year=it.get("year"),
                    country=it.get("country"),
                    label=it.get("label"),
                    catno=it.get("catno"),
                    resource_url=it.get("resource_url"),
                    uri=it.get("uri"),
                )
            )
        except Exception:
            # Skip any weird/partial items rather than exploding the whole response
            continue

    best = c_models[0] if c_models else None

    return IdentifyResponse(
        source_image=src_url,
        vision=v,
        best_guess=best,
        candidates=c_models,
        notes=None,
    )
