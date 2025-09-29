"""
FastAPI application for identifying records from images.

This module exposes a single POST /identify endpoint that accepts an image
upload (JPG/PNG/WEBP) and returns the most likely Discogs release URL,
alternates, confidence, and information extracted from the label. It uses
OpenAI's GPT‑4o vision model to extract text and visual cues from the
uploaded image, builds Discogs‑focused search queries, and queries the
Google Programmable Search API to find matching releases. A simple
heuristic then ranks results based on visual similarity to the uploaded
thumbnail. Environment variables are used to configure API keys.

If ``app/vision_openai.py`` is present, its ``extract_from_image``
function will be used to extract structured metadata (artist, title,
catalogNo, etc.). Otherwise a fallback stub will be used.
"""

import os
import re
import base64
import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from openai import OpenAI

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Environment variables for API keys and search engine ID
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# Instantiate OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------------------------------------------------
# Metadata extraction helper
# -----------------------------------------------------------------------------
# Try to import extract_from_image from vision_openai. If the module isn't
# present, define a fallback stub that returns an empty dict.
try:
    from .vision_openai import extract_from_image  # type: ignore
except Exception:
    async def extract_from_image(*args, **kwargs) -> Dict[str, Any]:
        return {}

# -----------------------------------------------------------------------------
# FastAPI app and middleware
# -----------------------------------------------------------------------------
app = FastAPI(title="GrooveID – Vision→Discogs Resolver (app/main.py)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # consider restricting in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logger = logging.getLogger("grooveid_app")
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def img_bytes_to_data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    """Return a data URI for the given image bytes."""
    return f"data:{mime};base64," + base64.b64encode(image_bytes).decode()

def keep_discogs_release_links(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter search results to only Discogs release or master URLs and get thumbnails."""
    kept: List[Dict[str, Any]] = []
    for it in items or []:
        link = it.get("link", "") or ""
        if "discogs.com" in link and ("/release/" in link or "/master/" in link):
            thumb = None
            pagemap = it.get("pagemap") or {}
            imgs = pagemap.get("cse_image") or pagemap.get("cse_thumbnail") or []
            if imgs and isinstance(imgs, list):
                thumb = imgs[0].get("src")
            kept.append({"url": link, "title": it.get("title", ""), "thumb": thumb})
    seen: set[str] = set()
    uniq: List[Dict[str, Any]] = []
    for c in kept:
        if c["url"] not in seen:
            uniq.append(c)
            seen.add(c["url"])
    return uniq

async def google_search(query: str, num: int = 10) -> Dict[str, Any]:
    """Perform a Google Programmable Search and return the JSON response."""
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
    """
    Build Discogs-targeted search queries from the output of vision_extract.

    Heuristics extract artist names, titles, catalog numbers and phone numbers
    from the raw text. The returned list is ordered from most specific to
    least specific. Duplicate queries are removed.
    """
    queries: List[str] = []
    raw_text = (v.get("raw_text") or "").strip()
    guesses = [g.strip() for g in (v.get("guesses") or []) if isinstance(g, str) and g.strip()]
    seeds = [q.strip() for q in (v.get("queries") or []) if isinstance(q, str) and q.strip()]
    vis = (v.get("visual_description") or "").strip()

    lines: List[str] = [l.strip() for l in raw_text.splitlines() if l.strip()]
    text_joined = " ".join(lines)

    phone_re = re.compile(r'(?:\+?\d[\d\-\s]{6,}\d)')
    catalog_re = re.compile(r'\b([A-Z]{1,6}\s?-?\s?\d{2,6}[A-Z]?)\b')
    phones = phone_re.findall(text_joined)
    catalogs = [m.group(1).replace(" ", "") for m in catalog_re.finditer(text_joined)][:3]

    artist: Optional[str] = None
    title: Optional[str] = None
    for l in lines:
        if 2 <= len(l) <= 30 and re.fullmatch(r"[A-Z0-9][A-Z0-9\-\s&/]{2,}", l):
            ll = l.lower()
            if not any(x in ll for x in ("stereo", "mono", "records", "side", "made in", "rpm", "produced")):
                artist = re.sub(r"\s+", " ", l).strip()
                break
    for l in lines:
        if re.search(r'\b(ep|lp|mix|remix|remixes|vol\.?|volume)\b', l, re.IGNORECASE):
            title = re.sub(r"\s+", " ", l).strip()
            break
    if not title:
        for l in lines:
            ws = l.split()
            if 1 < len(ws) <= 5 and l == l.lower():
                title = l.strip()
                break

    if artist and title:
        queries.append(f'site:discogs.com "{artist}" "{title}"')
        queries.append(f'site:discogs.com "{title}" "{artist}"')
    for c in catalogs:
        queries.append(f'site:discogs.com "{c}"')
    for p in phones[:2]:
        queries.append(f'site:discogs.com "{p}"')
    for g in guesses[:6]:
        if g.lower().startswith("site:discogs.com"):
            queries.append(g)
        else:
            queries.append(f'site:discogs.com "{g}"')
    for s in seeds[:6]:
        if s.lower().startswith("site:discogs.com"):
            queries.append(s)
        else:
            queries.append(f'site:discogs.com "{s}"')
    if raw_text:
        queries.append(f'site:discogs.com "{raw_text[:120]}"')
    if not queries and vis:
        queries.append(f'site:discogs.com "{vis}"')

    seen: set[str] = set()
    uniq: List[str] = []
    for q in queries:
        qn = q.strip().lower()
        if qn and qn not in seen:
            seen.add(qn)
            uniq.append(q.strip())
    return uniq[:15]

async def vision_extract(image_bytes: bytes) -> Dict[str, Any]:
    """
    Use OpenAI Vision (gpt‑4o) in OCR mode to extract text and build search queries.

    The prompt instructs the model to act as an OCR engine, extracting
    printed text and inferring likely metadata (artist, title, catalog numbers,
    etc.). It returns a JSON object with ``raw_text``, ``visual_description``,
    ``queries`` and ``guesses``. If the call fails, empty fields are returned.
    """
    data_url = img_bytes_to_data_url(image_bytes)
    system = (
        "You are an OCR engine and record-label reader. Extract ALL visible text "
        "exactly as printed (letters, numbers, hyphens, symbols) from the record label "
        "image. Also infer likely artist, title, label, catalog numbers and other "
        "useful tokens if present. Build 2–6 Discogs-targeted search queries (for example "
        "using site:discogs.com plus artist/title or catalog numbers). Return strict JSON "
        "with keys: raw_text (string), visual_description (string), queries (array of strings), "
        "guesses (array of strings). No commentary."
    )
    msg = [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "input_text", "text": "Extract text and build Discogs-ready queries."},
            {"type": "input_image", "image_url": data_url},
        ]},
    ]
    try:
        logger.info("[vision] calling OpenAI (OCR mode)")
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            messages=msg,
            response_format={"type": "json_object"},
        )
        import json
        parsed = json.loads(resp.choices[0].message.content or "{}")
        parsed.setdefault("raw_text", "")
        parsed.setdefault("visual_description", "")
        parsed.setdefault("queries", [])
        parsed.setdefault("guesses", [])
        return parsed
    except Exception as e:
        logger.error("[vision] FAILED: %s\n%s", e, traceback.format_exc())
        return {"raw_text": "", "visual_description": "", "queries": [], "guesses": []}

async def score_similarity_with_vision(query_img_bytes: bytes, candidate_thumb_url: str) -> Optional[float]:
    """
    Ask OpenAI Vision to score visual similarity between the uploaded image
    and a candidate thumbnail. Returns a float in [0,1] or None on error.
    """
    if not candidate_thumb_url:
        return None
    try:
        data_url = img_bytes_to_data_url(query_img_bytes)
        messages = [
            {"role": "system", "content": "Score visual similarity between two images of a record (0.0 to 1.0). Return ONLY a number."},
            {"role": "user", "content": [
                {"type": "input_text", "text": "Compare these two images and return a single number 0.0–1.0."},
                {"type": "input_image", "image_url": data_url},
                {"type": "input_image", "image_url": candidate_thumb_url},
            ]},
        ]
        resp = client.chat.completions.create(
            model="gpt-4o",
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

# -----------------------------------------------------------------------------
# Pydantic model for responses
# -----------------------------------------------------------------------------
class IdentifyResponse(BaseModel):
    discogs_url: Optional[str] = None
    confidence: Optional[float] = None
    alternates: List[Dict[str, Any]] = []
    used_queries: List[str] = []
    vision_text: Optional[str] = None
    vision_description: Optional[str] = None

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    """Simple health check endpoint."""
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
) -> IdentifyResponse:
    """
    Identify a record from an uploaded image.

    The endpoint uses OpenAI Vision to extract text and build search queries,
    queries Google CSE to find Discogs releases, and optionally re-ranks
    candidates based on visual similarity. If metadata extraction helper is
    available, additional targeted queries based on artist, title, catalogNo and
    keywords are prepended.
    """
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(400, "Please upload a JPG/PNG/WEBP image.")
    image_bytes = await file.read()
    logger.info("[identify] file=%s (%s bytes)", file.filename, len(image_bytes))

    # Step 1: Vision extract
    v = await vision_extract(image_bytes)
    queries = build_queries_from_vision(v)
    used_queries: List[str] = []

    # Step 2: Metadata-based queries (if helper returns data)
    try:
        meta = await extract_from_image(image_b64=base64.b64encode(image_bytes).decode())
    except Exception:
        meta = {}
    meta_queries: List[str] = []
    artist = meta.get("artist") if isinstance(meta.get("artist"), str) else None
    title = meta.get("title") if isinstance(meta.get("title"), str) else None
    catalog = meta.get("catalogNo") if isinstance(meta.get("catalogNo"), str) else None
    if artist and title:
        meta_queries.append(f'site:discogs.com "{artist}" "{title}"')
    if catalog:
        meta_queries.append(f'site:discogs.com "{catalog}"')
    for kw in (meta.get("keywords") or [])[:3]:
        if isinstance(kw, str) and kw.strip():
            meta_queries.append(f'site:discogs.com "{kw.strip()}"')
    # Prepend metadata queries
    queries = meta_queries + queries

    # Step 3: Search Discogs via Google
    discogs_candidates: List[Dict[str, Any]] = []
    try:
        for q in queries:
            used_queries.append(q)
            data = await google_search(q, num=10)
            items = data.get("items", [])
            discogs_candidates += keep_discogs_release_links(items)
            if len(discogs_candidates) >= max_candidates:
                break
        # Fallback when nothing found
        if not discogs_candidates:
            fallback_texts: List[str] = []
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
    except HTTPException as he:
        logger.error("[identify] controlled error: %s", getattr(he, "detail", he))
        raise he
    except Exception as e:
        logger.error("[identify] search loop FAILED: %s\n%s", e, traceback.format_exc())
        raise HTTPException(502, "Search pipeline failed")

    if not discogs_candidates:
        return IdentifyResponse(
            discogs_url=None,
            confidence=None,
            alternates=[],
            used_queries=used_queries,
            vision_text=v.get("raw_text", ""),
            vision_description=v.get("visual_description", ""),
        )

    best_url = discogs_candidates[0]["url"]
    best_score = 0.86
    ranked = discogs_candidates
    if do_visual_check:
        scored: List[Tuple[Dict[str, Any], float]] = []
        for c in discogs_candidates:
            score = await score_similarity_with_vision(image_bytes, c.get("thumb"))
            scored.append((c, score if score is not None else 0.5))
        scored.sort(key=lambda x: x[1], reverse=True)
        ranked = [c for c, _ in scored]
        best_url, best_score = ranked[0]["url"], scored[0][1]
        logger.info("[identify] best by visual=%.3f %s", best_score, best_url)
    alternates = [
        {"url": c["url"], "title": c.get("title", ""), "thumb": c.get("thumb")}
        for c in ranked[1:3]
    ]
    return IdentifyResponse(
        discogs_url=best_url,
        confidence=float(best_score),
        alternates=alternates,
        used_queries=used_queries,
        vision_text=v.get("raw_text", ""),
        vision_description=v.get("visual_description", ""),
    )
