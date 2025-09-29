import os
import re
import base64
import logging
import traceback
from typing import List, Dict, Any, Optional

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
# Attempt to import the metadata extractor from vision_openai. In some
# deployment setups (e.g. when running under `app.main` on Render),
# ``vision_openai.py`` may not be available on the Python path. To avoid
# import errors crashing the service, wrap this in a try/except and fall
# back to a no-op extractor if the module is absent.
try:
    from vision_openai import extract_from_image  # metadata helper
except ImportError:
    async def extract_from_image(*args, **kwargs) -> Dict[str, Any]:
        """Fallback metadata extractor that returns an empty result.

        When ``vision_openai.py`` is not present in the runtime, this stub
        will be used to satisfy calls in the identify pipeline. It simply
        returns an empty dict, so no metadata-based queries are added.
        """
        return {}

# --- OpenAI (>=1.0 SDK) ---
from openai import OpenAI

# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")  # Programmable Search Engine ID (cx)

client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------------------------------------------------------------------
# FastAPI setup
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
# Global exception handler
# ------------------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    logger.error("[unhandled] %s\n%s", exc, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": f"Server error: {type(exc).__name__}: {str(exc)}"},
    )

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
            thumb = None
            pagemap = it.get("pagemap") or {}
            imgs = pagemap.get("cse_image") or pagemap.get("cse_thumbnail") or []
            if imgs and isinstance(imgs, list):
                thumb = imgs[0].get("src")
            kept.append({"url": link, "title": it.get("title", ""), "thumb": thumb})
    # de-dupe
    seen, uniq = set(), []
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
    """
    Given the raw output from ``vision_extract``, construct a list of
    Discogs-focused search queries. This function uses heuristics to
    recognise artist names, titles, catalog numbers, phone numbers and
    other identifiers in the extracted text, and wraps them in
    ``site:discogs.com"..."`` queries. It also incorporates any search
    terms suggested by the model and falls back to the visual
    description when no text is available.

    The goal is to prioritise precise identifiers (artist/title pairs,
    catalog numbers, phone numbers) before falling back to more general
    guesses or visual descriptions. Duplicate queries are removed while
    preserving order.
    """
    queries: List[str] = []

    raw_text = (v.get("raw_text") or "").strip()
    guesses = [g.strip() for g in (v.get("guesses") or []) if isinstance(g, str) and g.strip()]
    seeds = [q.strip() for q in (v.get("queries") or []) if isinstance(q, str) and q.strip()]
    vis = (v.get("visual_description") or "").strip()

    # Parse raw text into lines and join for regex searches
    lines: List[str] = [l.strip() for l in raw_text.splitlines() if l.strip()]
    text_joined = " ".join(lines)

    # Regex patterns for phone numbers and catalog numbers
    phone_re = re.compile(r'(?:\+?\d[\d\-\s]{6,}\d)')
    catalog_re = re.compile(r'\b([A-Z]{1,6}\s?-?\s?\d{2,6}[A-Z]?)\b')

    phones = phone_re.findall(text_joined)
    catalogs = [m.group(1).replace(" ", "") for m in catalog_re.finditer(text_joined)][:3]

    # Heuristic extraction of artist and title
    artist: Optional[str] = None
    title: Optional[str] = None
    for l in lines:
        # Candidate artist: uppercase or hyphenated, not generic words
        if 2 <= len(l) <= 30 and re.fullmatch(r"[A-Z0-9][A-Z0-9\-\s&/]{2,}", l):
            ll = l.lower()
            if not any(x in ll for x in ("stereo", "mono", "records", "side", "made in", "rpm", "produced")):
                artist = re.sub(r"\s+", " ", l).strip()
                break
    # Candidate title: lines containing EP/LP/mix/remix or lowercase lines
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

    # Build queries starting with strongest identifiers
    if artist and title:
        queries.append(f'site:discogs.com "{artist}" "{title}"')
        queries.append(f'site:discogs.com "{title}" "{artist}"')
    # Catalog numbers and phone numbers
    for c in catalogs:
        queries.append(f'site:discogs.com "{c}"')
    for p in phones[:2]:
        queries.append(f'site:discogs.com "{p}"')
    # Guesses from the model
    for g in guesses[:6]:
        if g.lower().startswith("site:discogs.com"):
            queries.append(g)
        else:
            queries.append(f'site:discogs.com "{g}"')
    # Seeds (model-suggested search queries)
    for s in seeds[:6]:
        if s.lower().startswith("site:discogs.com"):
            queries.append(s)
        else:
            queries.append(f'site:discogs.com "{s}"')
    # Raw text fallback (truncated to avoid huge queries)
    if raw_text:
        queries.append(f'site:discogs.com "{raw_text[:120]}"')
    # Final fallback: use visual description if nothing else
    if not queries and vis:
        queries.append(f'site:discogs.com "{vis}"')

    # Deduplicate while preserving order
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
    Use OpenAI Vision (gpt-4o) in OCR mode to extract as much text as possible
    from the record label and build a set of initial search queries. This
    function instructs the model to behave as an OCR engine and record
    metadata extractor, rather than a generic description model. It returns
    a dict with keys: ``raw_text`` (newline-separated text extracted from
    the label), ``visual_description`` (a short caption describing the
    label's appearance), ``queries`` (a list of search query strings
    derived by the model), and ``guesses`` (tokens like artist, title,
    catalog numbers, phone numbers, etc. that can be used to build
    additional queries).

    The prompt is tailored to emphasise exact text extraction and Discogs
    search preparation. A low temperature and JSON response format ensure
    deterministic output. If the call fails, the function returns empty
    fields so the pipeline can fall back on other strategies.
    """
    data_url = img_bytes_to_data_url(image_bytes)
    # System prompt: emphasise OCR and metadata extraction for Discogs
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
        # normalise keys
        parsed.setdefault("raw_text", "")
        parsed.setdefault("visual_description", "")
        parsed.setdefault("queries", [])
        parsed.setdefault("guesses", [])
        return parsed
    except Exception as e:
        logger.error("[vision] FAILED: %s\n%s", e, traceback.format_exc())
        return {"raw_text": "", "visual_description": "", "queries": [], "guesses": []}

async def score_similarity_with_vision(query_img_bytes: bytes, candidate_thumb_url: str) -> Optional[float]:
    if not candidate_thumb_url:
        return None
    try:
        data_url = img_bytes_to_data_url(query_img_bytes)
        messages = [
            {"role": "system", "content":
                "Score visual similarity between two images of a record (0.0 to 1.0). "
                "Return ONLY a number."},
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
    logger.info("[identify] file=%s (%s bytes)", file.filename, len(image_bytes))

    v = await vision_extract(image_bytes)
    queries = build_queries_from_vision(v)
    used_queries: List[str] = []
    logger.info("[identify] initial queries=%s", queries)

    # ----------------------------------------------------------------------
    # Metadata-based queries via vision_openai
    # ----------------------------------------------------------------------
    # In addition to the raw text and guesses returned by vision_extract(),
    # use the strict metadata extractor to pull artist, title, catalog
    # numbers, and free keywords from the label. If present, these values
    # are used to build Discogs-targeted search queries that often yield
    # direct hits even when OCR on the image sticker fails (e.g., small
    # stickers or white-label pressings). These metadata queries are
    # prepended to the query list so they are tried first.
    try:
        meta = await extract_from_image(image_b64=base64.b64encode(image_bytes).decode())
    except Exception:
        meta = {}
    meta_queries: List[str] = []
    artist = meta.get("artist") if isinstance(meta.get("artist"), str) else None
    title = meta.get("title") if isinstance(meta.get("title"), str) else None
    if artist and title:
        meta_queries.append(f'site:discogs.com "{artist}" "{title}"')
    catalog = meta.get("catalogNo") if isinstance(meta.get("catalogNo"), str) else None
    if catalog:
        meta_queries.append(f'site:discogs.com "{catalog}"')
    for kw in (meta.get("keywords") or [])[:3]:
        if isinstance(kw, str) and kw.strip():
            meta_queries.append(f'site:discogs.com "{kw.strip()}"')

    # Prepend metadata-based queries so they are attempted before OCR-derived
    # and guess-based queries
    queries = meta_queries + queries

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
                logger.info("[fallback] %d items for: %s", len(items), q)
                discogs_candidates += keep_discogs_release_links(items)
                if len(discogs_candidates) >= max_candidates:
                    break

        discogs_candidates = discogs_candidates[:max_candidates]
        logger.info("[identify] discogs candidates=%d", len(discogs_candidates))

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
        scored: List[tuple[Dict[str, Any], float]] = []
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
