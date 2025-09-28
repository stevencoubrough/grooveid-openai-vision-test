# app/services/discogs_client.py

from typing import Dict, Any, List, Optional
import os

import httpx

DISCOGS_TOKEN = os.getenv("DISCOGS_USER_TOKEN")
BASE = "https://api.discogs.com"

# Reasonable defaults for API I/O
_TIMEOUT = httpx.Timeout(12.0, connect=4.0)
_LIMITS = httpx.Limits(max_connections=20, max_keepalive_connections=20)


def _q_parts(*parts: Optional[str]) -> str:
    """Join non-empty parts into a single search string."""
    return " ".join(p.strip() for p in parts if p and p.strip())


async def search_candidates(
    artist: Optional[str],
    title: Optional[str],
    label: Optional[str],
    catno: Optional[str],
    keywords: List[str],
) -> List[Dict[str, Any]]:
    """
    Hit Discogs search in a few targeted passes and return a de-duplicated list of results.
    We do not rank here; just collect plausible candidates.
    """
    if not DISCOGS_TOKEN:
        # We return empty set instead of raising; upstream can continue gracefully
        return []

    headers = {
        "Authorization": f"Discogs token={DISCOGS_TOKEN}",
        "User-Agent": "GrooveID/1.0 (+https://grooveid.app)",
    }

    params_list: List[Dict[str, str]] = []

    # High-precision pass: catalog number + label
    if catno and label:
        params_list.append(
            {"type": "release", "q": _q_parts(catno, label), "per_page": "25"}
        )

    # Strong pass: artist + title
    if artist and title:
        params_list.append(
            {"type": "release", "q": _q_parts(artist, title), "per_page": "25"}
        )

    # Keywords fallback (limit length)
    if keywords:
        params_list.append(
            {"type": "release", "q": " ".join(keywords[:6]), "per_page": "25"}
        )

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

            # De-duplicate by Discogs id
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
    """
    Lightweight scoring: catno/label is strongest, then artist/title, then year/country.
    Returns the same list, sorted by score (desc).
    """
    a = (artist or "").lower()
    t = (title or "").lower()
    l = (label or "").lower()
    c = (catno or "").lower()
    y = (year or "").lower()
    co = (country or "").lower()

    def score(it: Dict[str, Any]) -> int:
        s = 0
        # Exact catno match
        it_cat = (it.get("catno") or "").lower()
        if c and it_cat == c:
            s += 40

        # Label contains
        it_labels = [(x or "").lower() for x in (it.get("label") or [])]
        if l and any(l in lab for lab in it_labels):
            s += 20

        # Title string contains artist/title tokens
        it_title = (it.get("title") or "").lower()
        if a and a in it_title:
            s += 15
        if t and t in it_title:
            s += 15

        # Year equals
        it_year = str(it.get("year") or "").lower()
        if y and it_year == y:
            s += 6

        # Country equals
        it_country = (it.get("country") or "").lower()
        if co and it_country == co:
            s += 4

        return s

    return sorted(cands, key=score, reverse=True)
