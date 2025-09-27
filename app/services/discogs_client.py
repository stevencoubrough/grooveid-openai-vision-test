import os
from typing import Dict, Any, List, Optional, Tuple, Union
from urllib.parse import urlencode

from ..utils.http import get_client

DISCOGS_TOKEN = os.getenv("DISCOGS_USER_TOKEN")

BASE = "https://api.discogs.com"
HEADERS = {
    "Authorization": f"Discogs token={DISCOGS_TOKEN}",
    "User-Agent": "GrooveID/1.0 (+https://grooveid.app)"
}

def _q(*parts: Optional[str]) -> str:
    return " ".join(p for p in parts if p)

async def search_candidates(artist: Optional[str], title: Optional[str], label: Optional[str], catno: Optional[str], keywords: List[str]) -> List[Dict[str, Any]]:
    """
    Search Discogs for release candidates using a tiered strategy.
      1) catno + label (highest precision)
      2) artist + title
      3) keywords fallback (joined)
    Results are de-duplicated.
    """
    queries: List[Tuple[str, Dict[str, str]]] = []

    if catno and label:
        queries.append((_q(catno, label), {}))
    if artist and title:
        queries.append((_q(artist, title), {}))
    if keywords:
        q = " ".join(keywords[:6])
        queries.append((q, {}))

    results: List[Dict[str, Any]] = []
    async with get_client() as http:
        for q, extra in queries:
            params = {
                "q": q,
                "type": "release",
                "per_page": "25",
                **extra
            }
            url = f"{BASE}/database/search?{urlencode(params)}"
            r = await http.get(url, headers=HEADERS)
            if r.status_code == 401:
                raise RuntimeError("Discogs auth failed (check DISCOGS_USER_TOKEN).")
            r.raise_for_status()
            payload = r.json()
            items = payload.get("results", [])
            results.extend(items)

    # De-dupe by id while preserving order
    seen = set()
    unique: List[Dict[str, Any]] = []
    for it in results:
        _id = it.get("id")
        if _id in seen:
            continue
        seen.add(_id)
        unique.append(it)
    return unique

def rank_candidates(
    cands: List[Dict[str, Any]],
    artist: Optional[str],
    title: Optional[str],
    label: Optional[str],
    catno: Optional[str],
    year: Optional[str],
    country: Optional[str],
    return_scores: bool = False
) -> List[Union[Dict[str, Any], Tuple[int, Dict[str, Any]]]]:
    """
    Rank candidates using a simple scoring heuristic:
      +40 for exact catno match
      +20 if label appears in release labels
      +15 if artist appears in title
      +15 if title appears in title
      +6 if year matches
      +4 if country matches

    If return_scores is True, returns a list of (score, candidate) tuples instead of plain candidates.
    """
    def norm(s: Optional[str]) -> str:
        return (s or "").lower().strip()

    na, nt, nl, nc = map(norm, [artist, title, label, catno])
    ny, ncountry = norm(year), norm(country)

    def score(it: Dict[str, Any]) -> int:
        sc = 0
        if nc and norm(it.get("catno")) == nc:
            sc += 40
        lbls = [l.lower() for l in (it.get("label") or [])]
        if nl and any(nl in l for l in lbls):
            sc += 20
        t = norm(it.get("title"))
        if na and na in t:
            sc += 15
        if nt and nt in t:
            sc += 15
        if ny and str(it.get("year", "")).lower() == ny:
            sc += 6
        if ncountry and norm(it.get("country")) == ncountry:
            sc += 4
        return sc

    sorted_cands = sorted(cands, key=score, reverse=True)
    if return_scores:
        return [(score(it), it) for it in sorted_cands]
    else:
        return sorted_cands  # type: ignore
