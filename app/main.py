import base64
import os
import logging
from typing import Optional, List, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
# from pydantic import HttpUrl  # no longer needed
from dotenv import load_dotenv

from .models.schemas import IdentifyResponse, VisionExtract, DiscogsResult
from .services.vision_openai import extract_from_image
#  urllib.parse import urlparse
from .services.discogs_client import search_candidates, rank_candidates
from urllib.parse import urlparse


load_dotenv()

app = FastAPI(title="GrooveID Identify API", version="0.2.0")

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Feature flags
ENABLE_DEBUG = os.getenv("ENABLE_DEBUG", "false").lower() == "true"
LOG_SCORE_BREAKDOWN = os.getenv("LOG_SCORE_BREAKDOWN", "false").lower() == "true"


def _b64(file_bytes: bytes) -> str:
    return base64.b64encode(file_bytes).decode("utf-8")

# Last raw vision payload for debug route
last_vision_raw: Optional[dict] = None

@app.post("/identify", response_model=IdentifyResponse)
async def identify(
    image_url: Optional[str] = Form(None),
    image_file: Optional[UploadFile] = File(None),
) -> IdentifyResponse:
    """
    
        # Normalize and validate the URL; strip and check scheme/netloc
    raw_url = (image_url or "").strip()
    url = None
    if raw_url:
        parsed = urlparse(raw_url)
        if parsed.scheme and parsed.netloc:
            url = raw_url
    # End URL normalization
Identify a record by image URL or uploaded file. Returns structured data and ranked Discogs results.
    """

    # Normalize the URL: treat blank or whitespace strings as None
  #  url = (image_url or "").strip() if image_url else None

    if not url and not image_file:
        raise HTTPException(400, "Provide either image_url or image_file")

    b64 = None
    src_url = None

    if url:
        src_url = url
    else:
        # read the upload and convert to base64
        content = await image_file.read()
        if not content:
            raise HTTPException(400, "Empty file")
        b64 = _b64(content)

    # 1) Vision extraction
    try:
        vision = await extract_from_image(image_url=src_url, image_b64=b64)
    except Exception as e:
        logger.error("Vision extraction failed", exc_info=True)
        raise HTTPException(status_code=502, detail=f"Vision provider error: {e}")

    v = vision.data

    # store raw for debug
    global last_vision_raw
    last_vision_raw = v

    # 2) Discogs candidates
    artist = v.get("artist")
    title = v.get("title")
    label = v.get("label")
    catno = v.get("catalogNo")
    year = v.get("year")
    country = v.get("country")
    keywords = v.get("keywords") or []

    candidates = await search_candidates(artist, title, label, catno, keywords)
    ranked = rank_candidates(candidates, artist, title, label, catno, year, country)

    # Build response models
    c_models: List[DiscogsResult] = []
    for item in ranked:
        if isinstance(item, Tuple):
            score, it = item
            if LOG_SCORE_BREAKDOWN:
                logger.info("Candidate ID %s scored %s", it.get("id"), score)
        else:
            it = item

        model = DiscogsResult(
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
        c_models.append(model)

    best = c_models[0] if c_models else None

    # 3) Final JSON
    return IdentifyResponse(
        source_image=src_url,
        vision=VisionExtract(**v),
        best_guess=best,
        candidates=c_models,
        notes="Heuristic ranking: catno/label > artist/title > year > country",
    )

@app.get("/debug/echo")
async def debug_echo():
    if not ENABLE_DEBUG:
        raise HTTPException(status_code=404, detail="Debug disabled")
    return JSONResponse(last_vision_raw or {"message": "No vision data yet"})
