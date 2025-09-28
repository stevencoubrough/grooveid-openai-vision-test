import io
import os
import re
import base64
from typing import List, Dict, Any, Optional

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")  # your Programmable Search Engine ID

app = FastAPI(title="GrooveID – Vision→Discogs Resolver")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        raise HTTPException(500, "Google CSE is not configured.")
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_ID, "q": query, "num": num}
    async with httpx.AsyncClient(timeout=20) as http:
        r = await http.get(url, params=params)
    r.raise_for_status()
    return r.json()

def build_queries_from_vision(v: Dict[str, Any]) -> List[str]:
    queries = []
    if v.get("raw_text"):
        queries.append(v["raw_text"])
        queries.append(f"{v['raw_text']} vinyl")
    for q in v.get("queries", []):
        if isinstance(q, str) and q.strip():
            queries.append(q)
    for g in v.get("guesses", []):
        if isinstance(g, str) and g.strip():
            queries.append(f'site:discogs.com "{g.strip()}"')
    if v.get("visual_description"):
        queries.append(f'site:discogs.com {v["visual_description"]}')
    uniq, seen = [], set()
    for q in queries:
        q = q.strip()
        if q and q.lower() not in seen:
            uniq.append(q)
            seen.add(q.lower())
    return uniq[:15]

async def vision_extract(image_bytes: bytes) -> Dict[str, Any]:
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
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=msg,
        response_format={"type": "json_object"},
    )
    import json
    try:
        parsed = json.loads(resp.choices[0].message.content)
        parsed.setdefault("raw_text", "")
        parsed.setdefault("visual_description", "")
        parsed.setdefault("queries", [])
        parsed.setdefault("guesses", [])
        return parsed
    except Exception:
        return {"raw_text": "", "visual_description": "", "queries": [], "guesses": []}

class IdentifyResponse(BaseModel):
    discogs_url: Optional[str] = None
    alternates: List[Dict[str, Any]] = []
    used_queries: List[str] = []
    vision_text: Optional[str] = None
    vision_description: Optional[str] = None

# ---------------------------
# Endpoint
# ---------------------------

@app.post("/identify", response_model=IdentifyResponse)
async def identify(file: UploadFile = File(...), max_candidates: int = 8):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(400, "Please upload a JPG/PNG/WEBP image.")

    image_bytes = await file.read()

    # A) Vision
    v = await vision_extract(image_bytes)
    queries = build_queries_from_vision(v)
    used_queries = []

    # B) Google search
    discogs_candidates: List[Dict[str, Any]] = []
    for q in queries:
        used_queries.append(q)
        try:
            data = await google_search(q, num=10)

            # --- Debug print: show Google results ---
            print(f"\n[Google results for query: {q}]\n")
            for i, it in enumerate(data.get("items", []), 1):
                print(f"{i}. {it.get('title')} — {it.get('link')}")
            print("------------------------------------------------")

            items = data.get("items", [])
            discogs_candidates += keep_discogs_release_links(items)
            if len(discogs_candidates) >= max_candidates:
                break
        except Exception as e:
            print(f"Google search error: {e}")
            continue

    discogs_candidates = discogs_candidates[:max_candidates]

    if not discogs_candidates:
        return IdentifyResponse(
            discogs_url=None,
            alternates=[],
            used_queries=used_queries,
            vision_text=v.get("raw_text"),
            vision_description=v.get("visual_description"),
        )

    best_url = discogs_candidates[0]["url"]
    alternates = [{"url": c["url"], "title": c.get("title", "")} for c in discogs_candidates[1:3]]

    return IdentifyResponse(
        discogs_url=best_url,
        alternates=alternates,
        used_queries=used_queries,
        vision_text=v.get("raw_text"),
        vision_description=v.get("visual_description"),
    )
