import base64
import os
import logging
from typing import Optional, List, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import HttpUrl
from dotenv import load_dotenv

from .models.schemas import IdentifyResponse, VisionExtract, DiscogsResult
from .services.vision_openai import extract_from_image
from .services.discogs_client import search_candidates, rank_candidates

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
    image_url: Optional[HttpUrl] = Form(None),
    image_file: Optional[UploadFile] = File(None)
) -> IdentifyResponse:
    """
    Identify a record by image URL or uploaded file. Returns structured data and ranked Discogs results.
    """
    global last_vision_raw
    if not image_url and not image_file:
        raise HTTPException(400, "Provide either image_url or image_file")

    b64 = None
    src_url = None

    if image_url:
        src_url = str(image_url)
    else:
        # read the upload and convert to base64
        content = await image_file.read()
        if len(content) == 0:
            raise HTTPException(400, "Empty file")
        b64 = _b64(content)

    # 1) Vision extraction
    vision = await extract_from_image(image_url=src_url, image_b64=b64)
    v = vision.data

    # store raw for debug
    last_vision_raw = v

    # 2) Discogs candidates
    artist = v.get("artist")
    title = v.get("title")
    label = v.get("label")
    catno = v.get("catalogNo")
    year = v.get("year")
    country = v.get("country")
    keywords = v.get("keywords", []) or []

    candidates = await search_candidates(artist, title, label, catno, keywords)
    ranked = rank_candidates(candidates, artist, title, label, catno, year, country, return_scores=LOG_SCORE_BREAKDOWN)

    # Build response models
    c_models: List[DiscogsResult] = []
    best: Optional[DiscogsResult] = None

    for rank in ranked:
        # rank may be (score, item) if LOG_SCORE_BREAKDOWN else item
        if isinstance(rank, tuple):
            score, it = rank  # type: ignore
            if LOG_SCORE_BREAKDOWN:
                logger.info("Candidate ID %s scored %s", it.get("id"), score)
        else:
            it = rank

        model = DiscogsResult(
            id=it.get("id"),
            type=it.get("type"),
            title=it.get("title"),
            year=it.get("year"),
            country=it.get("country"),
            label=it.get("label"),
            catno=it.get("catno"),
            resource_url=it.get("resource_url"),
            uri=it.get("uri")
        )
        c_models.append(model)

    if c_models:
        best = c_models[0]

    # 3) Final JSON
    return IdentifyResponse(
        source_image=src_url,
        vision=VisionExtract(**v),
        best_guess=best,
        candidates=c_models,
        notes="Heuristic ranking: catno/label > artist/title > year > country"
    )

@app.get("/debug/echo")
async def debug_echo() -> JSONResponse:
    """
    Return the last raw vision extraction JSON. Enabled only when ENABLE_DEBUG=true.
    """
    if not ENABLE_DEBUG:
        raise HTTPException(404, "Debug endpoint disabled.")
    if last_vision_raw is None:
        return JSONResponse(content={"error": "No vision extraction performed yet."})
    return JSONResponse(content=last_vision_raw)

@app.get("/", response_class=FileResponse)
async def index() -> FileResponse:
    """
    Serve a simple front-end page for testing the API.
    """
    static_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    return FileResponse(static_path, media_type="text/html")
