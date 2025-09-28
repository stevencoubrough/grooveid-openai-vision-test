# app/services/discogs_client.py

from typing import Dict, Any, List, Optional
import os
import httpx

DISCOGS_TOKEN = os.getenv("DISCOGS_USER_TOKEN")
BASE = "https://api.discogs.com"

_TIMEOUT = httpx.Timeout(12.0, connect=4.0)
_LIMITS = httpx.Limits(max_connections=20, max_keepalive_connections=20)

def _q_parts(*parts: Optional[str]) -> str:
    return " ".join(p.strip() for p in parts if p and p.strip())

async def search_candidates(
    artist: Optional[str],
    title: Optional[str],
    label: Optional[str],
    catno: Optional[str],
    keywords: List[str],
) -> List[Dict[str, Any]]:
    if not DISCOGS_TOKEN:
        return []

    headers = {
        "Authorization": f"Discogs token={DISCOGS_TOKEN}",
        "User-Agent": "GrooveID/1.0 (+https://grooveid.app)",
    }

    params_list: List[Dict[str, str]] = []
    if catno and label:
        params_list.append({"type": "release", "q": _q_parts(catno, label), "per_page": "25"})
    if artist and title:
        params_list.append({"type": "release", "q": _q_parts(artist, title), "per_page": "25"})
    if keywords:
        params_list.append({"type": "release", "q": " ".join(keywords[:6]), "per_page": "25"})

    results: List[Dict[str, Any]] = []
    seen = set()

    async with httpx.AsyncClient(timeout=_TIMEOUT, limits=_LIMITS, headers=headers) as http:
        for params in params_list:
            url = f"{BASE}/database/search"
            try:
                r = await http.get(url, params=params)
                r.raise_for_status()
                items = r.json().get("results", [])
            except Exception:
                items = []

            for it in items:
                _id = it.get("id")
                if _id in seen:
                    continue
                seen.add(_id)
                results.append(it)

    return results

def rank_candidates(
    cands: List[Dict[str, Any]],
    artist: Optional[str],
    title: Optional[str],
    label: Optional[str],
    catno: Optional[str],
    year: Optional[str],
    country: Optional[str],
) -> List[Dict[str, Any]]:
    a = (artist or "").lower()
    t = (title or "").lower()
    l = (label or "").lower()
    c = (catno or "").lower()
    y = (year or "").lower()
    co = (country or "").lower()

    def score(it: Dict[str, Any]) -> int:
        s = 0
        if c and (it.get("catno") or "").lower() == c:
            s += 40
        labels = [(x or "").lower() for x in (it.get("label") or [])]
        if l and any(l in lab for lab in labels):
            s += 20
        ttitle = (it.get("title") or "").lower()
        if a and a in ttitle:
            s += 15
        if t and t in ttitle:
            s += 15
        if y and str(it.get("year") or "").lower() == y:
            s += 6
        if co and (it.get("country") or "").lower() == co:
            s += 4
        return s

    return sorted(cands, key=score, reverse=True)
