import io
import os
import re
import base64
from typing import List, Dict, Any, Optional

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---- OpenAI (>=1.0 SDK) ----
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")  # your Programmable Search Engine ID

app = FastAPI(title="GrooveID – Vision→Discogs Resolver")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Helpers
# ---------------------------

def img_bytes_to_data_url(b: bytes) -> str:
    return f"data:image/jpeg;base64,{base64.b64encode(b).decode()}"

def keep_discogs_release_links(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept = []
    for it in items or []:
        link = it.get("link", "")
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
        raise HTTPException(500, "Google CSE is not configured (GOOGLE_API_KEY / GOOGLE_CSE_ID).")
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_ID, "q": query, "num": num}
    async with httpx.AsyncClient(timeout=20) as http:
        r = await http.get(url, params=params)
    r.raise_for_status()
    return r.json()

def build_queries_from_vision(vision_json: Dict[str, Any]) -> List[str]:
    """
    vision_json is what we ask the model to return: dict with keys:
    - raw_text (string from OCR if any)
    - visual_description (short)
    - queries (list of search strings)
    - guesses (list of plausible artist/label/title guesses)
    """
    queries = []
    raw_text = vision_json.get("raw_text") or ""
    if raw_text.strip():
        queries.append(raw_text.strip())
        # a more directed try:
        queries.append(f'{raw_text.strip()} vinyl')
    for q in vision_json.get("queries", [])[:10]:
        if isinstance(q, str):
            queries.append(q)
    # add a couple Discogs-biased queries if we got guesses
    for g in vision_json.get("guesses", [])[:5]:
        if isinstance(g, str) and g.strip():
            queries.append(f'site:discogs.com "{g.strip()}"')
    # always add a generic Discogs bias using description
    vis = (vision_json.get("visual_description") or "").strip()
    if vis:
        queries.append(f'site:discogs.com {vis}')
    # keep them unique & non-empty
    uniq = []
    seen = set()
    for q in queries:
        q = q.strip()
        if q and q.lower() not in seen:
            uniq.append(q)
            seen.add(q.lower())
    return uniq[:15]

async def vision_extract(image_bytes: bytes) -> Dict[str, Any]:
    """
    Ask Vision to:
      - OCR any visible text
      - If little/no text, describe visuals & propose queries & plausible guesses
    """
    data_url = img_bytes_to_data_url(image_bytes)

    system = (
        "You are a record identifier assistant. "
        "Given a single album/label photo, do three things:\n"
        "1) OCR any visible text (artist, title, label, catalog#). If none, return an empty string.\n"
        "2) Provide a concise visual description (colors, shapes, layout, distinguishing marks).\n"
        "3) Propose 8–12 web search queries that could find the Discogs page for this record, mixing:\n"
        "   - exact text you can read (if any),\n"
        "   - descriptive keywords (e.g., 'plain olive label tiny red figure'),\n"
        "   - a few plausible proper-noun guesses if confident (artist/label/series).\n"
        "4) Provide up to 5 'guesses' (artist/label/title) if you have plausible ideas; else empty list.\n"
        "Return strict JSON with keys: raw_text, visual_description, queries, guesses."
    )

    msg = [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "input_text", "text": "Identify and propose search queries for Discogs."},
            {"type": "input_image", "image_url": data_url},
        ]},
    ]

    # Using the Responses API if available; otherwise fall back to chat.completions.
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # vision-capable, swap to your deployed model
            temperature=0.2,
            messages=msg,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
    except Exception as e:
        raise HTTPException(500, f"OpenAI Vision error: {e}")

    import json
    try:
        parsed = json.loads(content)
        # basic shape guard
        parsed.setdefault("raw_text", "")
        parsed.setdefault("visual_description", "")
        parsed.setdefault("queries", [])
        parsed.setdefault("guesses", [])
        return parsed
    except Exception:
        # If model returned non-JSON for some reason, degrade gracefully.
        return {"raw_text": "", "visual_description": "", "queries": [], "guesses": []}

async def score_similarity_with_vision(query_img_bytes: bytes, candidate_thumb_url: str) -> Optional[float]:
    """
    Ask the model to score visual similarity between the scanned photo and a candidate thumbnail.
    Returns float in [0,1], or None if failed. Stateless (no storage).
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
        # extract first float-like token
        m = re.search(r"([01](?:\.\d+)?)", raw)
        if m:
            val = float(m.group(1))
            # clamp
            return max(0.0, min(1.0, val))
    except Exception:
        return None
    return None


class IdentifyResponse(BaseModel):
    discogs_url: Optional[str] = None
    confidence: Optional[float] = None
    alternates: List[Dict[str, Any]] = []
    used_queries: List[str] = []
    vision_text: Optional[str] = None
    vision_description: Optional[str] = None


# ---------------------------
# Endpoint
# ---------------------------

@app.post("/identify", response_model=IdentifyResponse)
async def identify(file: UploadFile = File(...), max_candidates: int = 8, do_visual_check: bool = True):
    """
    Photo in -> Discogs link out (stateless).
    - Uses OpenAI Vision to extract text / visual cues and propose queries
    - Google CSE to search the web
    - Keeps only discogs.com release/master pages
    - Optional: asks Vision to visually compare the scan to candidate thumbnails and score similarity
    """
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(400, "Please upload a JPG/PNG/WEBP image.")

    image_bytes = await file.read()

    # A) Vision: get OCR/visual descriptors + queries
    v = await vision_extract(image_bytes)
    queries = build_queries_from_vision(v)
    used_queries = []

    # B) Search pass 1: general (no site filter), keep only Discogs from results
    discogs_candidates: List[Dict[str, Any]] = []
    for q in queries:
        used_queries.append(q)
        try:
            data = await google_search(q, num=10)
            items = data.get("items", [])
            discogs_candidates += keep_discogs_release_links(items)
            if len(discogs_candidates) >= max_candidates:
                break
        except Exception:
            # ignore single-query errors; move on
            continue

    # C) If none found, do a Discogs-biased retry using the top 3 strongest text queries we have
    if not discogs_candidates:
        fallback_texts = []
        if v.get("raw_text"):
            fallback_texts.append(v["raw_text"])
        fallback_texts.extend(v.get("guesses", []) or [])
        # unique, non-empty, short
        fb = [t.strip() for t in fallback_texts if isinstance(t, str) and t.strip()]
        fb = fb[:3] or ["vinyl record minimal label"]
        for t in fb:
            q = f'site:discogs.com "{t}"'
            used_queries.append(q)
            try:
                data = await google_search(q, num=10)
                items = data.get("items", [])
                discogs_candidates += keep_discogs_release_links(items)
                if len(discogs_candidates) >= max_candidates:
                    break
            except Exception:
                continue

    # D) De-duplicate and cap to max_candidates
    # (already de-duped in helper, but cap again)
    discogs_candidates = discogs_candidates[:max_candidates]

    if not discogs_candidates:
        return IdentifyResponse(
            discogs_url=None,
            confidence=None,
            alternates=[],
            used_queries=used_queries,
            vision_text=v.get("raw_text", ""),
            vision_description=v.get("visual_description", ""),
        )

    # E) Optional: Visual similarity re-check (stateless)
    # If we have thumbnails, ask Vision to score and take the best.
    best_url = discogs_candidates[0]["url"]
    best_score = 0.86  # default if we don't re-check
    ranked = discogs_candidates

    if do_visual_check:
        scored = []
        for c in discogs_candidates:
            score = await score_similarity_with_vision(image_bytes, c.get("thumb"))
            # If no thumb or failure, set a modest default to preserve order
            scored.append((c, score if score is not None else 0.5))
        scored.sort(key=lambda x: x[1], reverse=True)
        ranked = [c for c, _ in scored]
        best_url, best_score = ranked[0]["url"], scored[0][1]

    # F) Build alternates (up to 2 others)
    alternates = []
    for c in ranked[1:3]:
        alternates.append({"url": c["url"], "title": c.get("title", ""), "thumb": c.get("thumb")})

    return IdentifyResponse(
        discogs_url=best_url,
        confidence=float(best_score),
        alternates=alternates,
        used_queries=used_queries,
        vision_text=v.get("raw_text", ""),
        vision_description=v.get("visual_description", ""),
    )
