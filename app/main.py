import os
import re
import base64
import logging
import traceback
from typing import List, Dict, Any, Optional

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- OpenAI (>=1.0 SDK) ---
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")  # Programmable Search Engine ID (cx)

client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------------------------------------------------------------------
# App
# ------------------------------------------------------------------------------
app = FastAPI(title="GrooveID – Vision→Discogs Resolver")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logger = logging.getLogger("grooveid")
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def img_bytes_to_data_url(b: bytes) -> str:
    return f"data:image/jpeg;base64,{base64.b64encode(b).decode()}"

def keep_discogs_release_links(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept = []
    for it in items or []:
        link = it.get("link", "") or ""
        if "discogs.com" in link and ("/release/" in link or "/master/" in link):
            # Try to grab a thumbnail from CSE payload if present
            thumb = None
            pagemap = it.get("pagemap") or {}
            imgs = pagemap.get("cse_image") or pagemap.get("cse_thumbnail") or []
            if imgs and isinstance(imgs, list):
                thumb = imgs[0].get("src")
            kept.append({"url": link, "title": it.get("title", ""), "thumb": thumb})
    # de-dupe by url
    seen = set()
    uniq = []
    for c in kept:
        if c["url"] not in seen:
            uniq.append(c)
            seen.add(c["url"])
    return uniq

async def google_search(query: str, num: int = 10) -> Dict[str, Any]:
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        raise HTTPException(500, "Google CSE not configured")
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_ID, "q": query, "num": num}
    try:
        logger.info("[google] %s", query)
        async with httpx.AsyncClient(timeout=20) as http:
            r = await http.get(url, params=params)
        if r.status_code >= 400:
            logger.error("[google] HTTP %s: %s", r.status_code, r.text[:500])
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error("[google] FAILED: %s\n%s", e, traceback.format_exc())
        raise HTTPException(502, f"Google search error for '{query}'")

def build_queries_from_vision(v: Dict[str, Any]) -> List[str]:
    queries: List[str] = []

    raw_text = (v.get("raw_text") or "").strip()
    if raw_text:
        queries.append(raw_text)
        queries.append(f"{raw_text} vinyl")

    # model-suggested queries
    for q in v.get("queries", []):
        if isinstance(q, str) and q.strip():
            queries.append(q.strip())

    # guesses become Discogs-biased queries
    for g in v.get("guesses", []):
        if isinstance(g, str) and g.strip():
            queries.append(f'site:discogs.com "{g.strip()}"')

    # always add a Discogs-biased descriptive query
    vis = (v.get("visual_description") or "").strip()
    if vis:
        queries.append(f"site:discogs.com {vis}")

    # unique + cap
    seen = set()
    uniq = []
    for q in queries:
        qn = q.lower()
        if qn and qn not in seen:
            seen.add(qn)
            uniq.append(q)
    return uniq[:15]

async def vision_extract(image_bytes: bytes) -> Dict[str, Any]:
    """
    Ask Vision to OCR & describe; return:
      { raw_text, visual_description, queries[], guesses[] }
    """
    data_url = img_bytes_to_data_url(image_bytes)
    system = (
        "You are a record identifier assistant. "
        "Return JSON with keys: raw_text, visual_description, queries, guesses."
    )
    msg = [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "input_text", "text": "Identify and propose search queries for Discogs."},
            {"type": "input_image", "image_url": data_url},
        ]},
    ]
    try:
        logger.info("[vision] calling OpenAI")
        # If your key lacks access to gpt-4o-mini, try "gpt-4o" or "gpt-4.1-mini"
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=msg,
            response_format={"type": "json_object"},
        )
        import json
        parsed = json.loads(resp.choices[0].message.content)
        parsed.setdefault("raw_text", "")
        parsed.setdefault("visual_description", "")
        parsed.setdefault("queries", [])
        parsed.setdefault("guesses", [])
        logger.info("[vision] ok")
        return parsed
    except Exception as e:
        logger.error("[vision] FAILED: %s\n%s", e, traceback.format_exc())
        # degrade gracefully so the endpoint doesn’t 500
        return {"raw_text": "", "visual_description": "", "queries": [], "guesses": []}

async def score_similarity_with_vision(query_img_bytes: bytes, candidate_thumb_url: str) -> Optional[float]:
    """
    Ask the model to score similarity between uploaded photo and candidate thumbnail.
    Returns 0..1 or None if not comparable. Stateless (no storage).
    """
    if not candidate_thumb_url:
        return None
    try:
        data_url = img_bytes_to_data_url(query_img_bytes)
        messages = [
            {"role": "system", "content":
                "Score visual similarity between two images of a record (0.0 to 1.0). "
                "Consider colors, layout, motifs, and distinctive marks. Return ONLY a number."},
            {"role": "user", "content": [
                {"type": "input_text", "text": "Compare these two images and return a single number 0.0–1.0."},
                {"type": "input_image", "image_url": data_url},
                {"type": "input_image", "image_url": candidate_thumb_url},
            ]},
        ]
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=messages,
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r"([01](?:\.\d+)?)", raw)
        if m:
            val = float(m.group(1))
            return max(0.0, min(1.0, val))
    except Exception as e:
        logger.warning("[similarity] FAILED for %s: %s", candidate_thumb_url, e)
    return None

# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------
class IdentifyResponse(BaseModel):
    discogs_url: Optional[str] = None
    confidence: Optional[float] = None
    alternates: List[Dict[str, Any]] = []
    used_queries: List[str] = []
    vision_text: Optional[str] = None
    vision_description: Optional[str] = None

# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "has_openai_key": bool(OPENAI_API_KEY),
        "has_google_key": bool(GOOGLE_API_KEY),
        "has_google_cx": bool(GOOGLE_CSE_ID),
    }

@app.post("/identify", response_model=IdentifyResponse)
async def identify(
    file: UploadFile = File(...),
    max_candidates: int = Query(8, ge=1, le=20),
    do_visual_check: bool = Query(True),
):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(400, "Please upload a JPG/PNG/WEBP image.")
    image_bytes = await file.read()

    # A) Vision
    v = await vision_extract(image_bytes)
    queries = build_queries_from_vision(v)
    used_queries: List[str] = []

    # B) Search pass (general; keep only Discogs)
    try:
        discogs_candidates: List[Dict[str, Any]] = []
        for q in queries:
            used_queries.append(q)
            data = await google_search(q, num=10)
            items = data.get("items", [])
            logger.info("[results] %d items for: %s", len(items), q)
            for i, it in enumerate(items[:5], 1):
                logger.info("  %d) %s — %s", i, it.get("title"), it.get("link"))
            discogs_candidates += keep_discogs_release_links(items)
            if len(discogs_candidates) >= max_candidates:
                break

        # Fallback: if none found, retry with guesses/raw_text Discogs-biased
        if not discogs_candidates:
            fallback_texts = []
            if v.get("raw_text"):
                fallback_texts.append(v["raw_text"])
            fallback_texts.extend(v.get("guesses", []) or [])
            fb = [t.strip() for t in fallback_texts if isinstance(t, str) and t.strip()]
            fb = fb[:3] or ["vinyl record minimal label"]
            for t in fb:
                q = f'site:discogs.com "{t}"'
                used_queries.append(q)
                data = await google_search(q, num=10)
                items = data.get("items", [])
                discogs_candidates += keep_discogs_release_links(items)
                if len(discogs_candidates) >= max_candidates:
                    break

        discogs_candidates = discogs_candidates[:max_candidates]
    except HTTPException:
        # forward controlled HTTP erro
