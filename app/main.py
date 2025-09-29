# app/main.py
import os, io, re, json, base64, traceback
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
from PIL import Image, ImageOps, ImageFilter

# --- OpenAI client -----------------------------------------------------------
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- FastAPI -----------------------------------------------------------------
app = FastAPI(title="GrooveID Identify API", version="1.0.0")

# --- Config ------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID  = os.getenv("GOOGLE_CSE_ID", "")

# --- Utilities ---------------------------------------------------------------
def img_bytes_to_data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64," + base64.b64encode(image_bytes).decode()

def pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

def bytes_from_pil(img: Image.Image, fmt: str = "JPEG", quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return buf.getvalue()

def upscale(img: Image.Image, min_w: int = 1400) -> Image.Image:
    if img.width >= min_w:
        return img
    scale = min_w / float(img.width)
    return img.resize((int(img.width*scale), int(img.height*scale)), Image.LANCZOS)

def micro_crops(img: Image.Image) -> List[Image.Image]:
    """
    Generate a few likely regions where small sticker text lives:
      - center disc area
      - lower quadrant (often sticker)
      - full image sharpened
    """
    W, H = img.size
    crops: List[Image.Image] = []

    # Center square
    s = int(min(W, H) * 0.55)
    cx, cy = W//2, H//2
    center = img.crop((cx - s//2, cy - s//2, cx + s//2, cy + s//2))
    crops.append(center)

    # Lower band
    lh = int(H * 0.35)
    lower = img.crop((int(W*0.1), H - lh, int(W*0.9), H))
    crops.append(lower)

    # Slight left/right bottom wedges
    bl = img.crop((0, int(H*0.55), int(W*0.6), H))
    br = img.crop((int(W*0.4), int(H*0.55), W, H))
    crops += [bl, br]

    # Add a globally sharpened version
    sharp = img.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=2))
    crops.append(sharp)

    return [upscale(c) for c in crops]

def preprocess_ocr(img: Image.Image) -> Image.Image:
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g, cutoff=1)
    g = g.filter(ImageFilter.MedianFilter(size=3))
    g = g.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=2))
    # gentle threshold to make faint sticker text pop
    g = g.point(lambda p: 255 if p > 175 else (0 if p < 120 else p))
    return g

def force_json(resp_content: str) -> Dict[str, Any]:
    try:
        return json.loads(resp_content or "{}")
    except Exception:
        return {}

# --- OpenAI Vision calls -----------------------------------------------------
def openai_ocr_json(image_bytes: bytes) -> Dict[str, Any]:
    """
    Ask OpenAI for STRICT OCR (no commentary) and Discogs-ready tokens.
    Returns keys: raw_text, visual_description, queries, guesses
    """
    data_url = img_bytes_to_data_url(image_bytes)
    system = (
        "You are an OCR engine and record-label reader. "
        "Extract ALL visible text exactly as printed (letters, numbers, hyphens, symbols). "
        "Also infer likely artist/title/label/catalog numbers if present. "
        "Return strict JSON with keys:\n"
        "  - raw_text: string (all text, newline-separated)\n"
        "  - visual_description: short caption of the label\n"
        "  - queries: array of 2-8 Discogs-targeted search queries (strings) built from text\n"
        "  - guesses: array of tokens (artist/title/label/catalog/phones/years)\n"
        "No commentary."
    )
    messages = [
        {"role":"system","content":system},
        {"role":"user","content":[
            {"type":"input_text","text":"Extract text and build Discogs-ready queries."},
            {"type":"input_image","image_url":data_url}
        ]}
    ]
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            response_format={"type":"json_object"},
            messages=messages,
        )
        return force_json(resp.choices[0].message.content)
    except Exception:
        return {}

def openai_visual_keywords_json(image_bytes: bytes) -> Dict[str, Any]:
    """
    Ask OpenAI for compact, search-friendly visual tokens (NOT prose).
    Returns keys: visual_description, queries, guesses
    """
    data_url = img_bytes_to_data_url(image_bytes)
    system = (
        "You are an image keyword generator for record labels. "
        "When no readable text exists, output compact search tokens: label features, colors, logos/letters you think you see, "
        "format cues (12\", EP, promo, test pressing), genre hints, country, approximate year range. "
        "Return strict JSON with keys:\n"
        "  - visual_description: short caption of the label\n"
        "  - queries: array of 4-10 Discogs-targeted search queries (strings)\n"
        "  - guesses: array of tokens (e.g., 'white label', 'hand-stamped', 'tech house', 'UK', '1996..2000')\n"
        "No commentary."
    )
    messages = [
        {"role":"system","content":system},
        {"role":"user","content":[
            {"type":"input_text","text":"Generate Discogs search tokens and queries from visuals only."},
            {"type":"input_image","image_url":data_url}
        ]}
    ]
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            response_format={"type":"json_object"},
            messages=messages,
        )
        return force_json(resp.choices[0].message.content)
    except Exception:
        return {}

# --- Google Programmable Search ---------------------------------------------
async def google_cse_search(query: str) -> List[Dict[str, Any]]:
    """
    Returns a list of results with: link, title, thumbnail
    """
    if not (GOOGLE_API_KEY and GOOGLE_CSE_ID):
        return []
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_ID, "q": query, "num": 10}
    try:
        async with httpx.AsyncClient(timeout=15) as s:
            r = await s.get(url, params=params)
            if r.status_code != 200:
                return []
            js = r.json()
            items = js.get("items") or []
            out = []
            for it in items:
                link = it.get("link", "")
                title = it.get("title", "")
                thumb = ""
                pagemap = it.get("pagemap") or {}
                # try to pull a thumbnail (Discogs often exposes via cse_image)
                if "cse_image" in pagemap and pagemap["cse_image"]:
                    thumb = pagemap["cse_image"][0].get("src","") or ""
                out.append({"link": link, "title": title, "thumbnail": thumb})
            return out
    except Exception:
        return []

def filter_discogs(results: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    Keep only Discogs /release or /master links, return list of (url, thumb)
    """
    out = []
    for r in results:
        url = (r.get("link") or "").lower()
        if "discogs.com" in url and ("/release/" in url or "/master/" in url):
            out.append((r.get("link"), r.get("thumbnail") or ""))
    # de-dup preserving order
    seen, uniq = set(), []
    for u,t in out:
        if u not in seen:
            uniq.append((u,t)); seen.add(u)
    return uniq

# --- Query builders ----------------------------------------------------------
PHONE_RE   = re.compile(r'(?:\+?\d[\d\-\s]{6,}\d)')
CATALOG_RE = re.compile(r'\b([A-Z]{1,6}\s?-?\s?\d{2,6}[A-Z]?)\b')

def build_discogs_queries_from_text(raw_text: str, guesses: List[str], visual_desc: str) -> List[str]:
    lines = [l.strip() for l in (raw_text or "").splitlines() if l.strip()]
    txt = " ".join(lines)
    phones = PHONE_RE.findall(txt)
    catalogs = [m.group(1).replace(" ", "") for m in CATALOG_RE.finditer(txt)][:3]

    # artist/title heuristics
    artist, title = None, None
    for l in lines:
        if 2 <= len(l) <= 30 and re.fullmatch(r"[A-Z0-9][A-Z0-9\-\s&/]{2,}", l):
            if not any(x in l.lower() for x in ("stereo","mono","records","side","made in","rpm")):
                artist = re.sub(r"\s+"," ", l); break
    for l in lines:
        if re.search(r'\b(ep|lp|mix|remix|remixes|vol\.?|volume)\b', l, re.I):
            title = re.sub(r"\s+"," ", l); break
    if not title:
        for l in lines:
            ws = l.split()
            if 1 < len(ws) <= 5 and l.lower() == l:
                title = l; break

    q: List[str] = []
    if artist and title:
        q += [
            f'site:discogs.com "{artist}" "{title}"',
            f'site:discogs.com "{title}" "{artist}"',
        ]
    for c in catalogs:
        q.append(f'site:discogs.com "{c}"')
    for p in phones[:2]:
        q.append(f'site:discogs.com "{p}"')

    # guesses from model (limit & quote smartly)
    for g in (guesses or [])[:6]:
        g = g.strip()
        if not g: 
            continue
        if g.lower().startswith("site:discogs.com"):
            q.append(g)
        else:
            q.append(f'site:discogs.com "{g}"')

    # raw text snippet (avoids too-long queries)
    if raw_text:
        q.append(f'site:discogs.com "{raw_text[:120]}"')
    if not q and visual_desc:
        q.append(f'site:discogs.com "{visual_desc}"')

    # dedupe
    seen=set(); uniq=[]
    for s in q:
        if s not in seen:
            uniq.append(s); seen.add(s)
    return uniq[:12]

def build_discogs_queries_from_visual(visual_json: Dict[str,Any]) -> List[str]:
    visual_desc = (visual_json.get("visual_description") or "").strip()
    guesses    = [g.strip() for g in (visual_json.get("guesses") or []) if isinstance(g, str)]
    seeds      = [q.strip() for q in (visual_json.get("queries") or []) if isinstance(q, str)]

    q: List[str] = []
    for s in seeds[:10]:
        if s.lower().startswith("site:discogs.com"):
            q.append(s)
        else:
            q.append(f'site:discogs.com "{s}"')
    # Add a couple mixed queries from guesses
    if guesses:
        q.append("site:discogs.com " + " ".join(f'"{g}"' for g in guesses[:3]))
    if visual_desc:
        q.append(f'site:discogs.com "{visual_desc}"')

    # dedupe
    seen=set(); uniq=[]
    for s in q:
        if s not in seen:
            uniq.append(s); seen.add(s)
    return uniq[:12]

# --- Optional: visual re-rank -----------------------------------------------
async def fetch_image_bytes(url: str) -> bytes:
    try:
        async with httpx.AsyncClient(timeout=10) as s:
            r = await s.get(url, follow_redirects=True)
            if r.status_code == 200 and r.content:
                return r.content
    except Exception:
        pass
    return b""

async def compare_visual_similarity(query_img: bytes, candidate_thumb_url: str) -> float:
    if not candidate_thumb_url:
        return 0.0
    cand = await fetch_image_bytes(candidate_thumb_url)
    if not cand:
        return 0.0

    def to_du(b): return "data:image/jpeg;base64," + base64.b64encode(b).decode()
    messages = [
        {"role":"system","content":"You are an image matcher. Score 0–1 how likely these two images show the SAME record label/pressing."},
        {"role":"user","content":[
            {"type":"input_text","text":'Return JSON: {"score": number}. No commentary.'},
            {"type":"input_image","image_url":to_du(query_img)},
            {"type":"input_image","image_url":to_du(cand)},
        ]}
    ]
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            response_format={"type":"json_object"},
            messages=messages,
        )
        js = force_json(resp.choices[0].message.content or "{}")
        s = float(js.get("score", 0))
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.0

# --- Core identify flow ------------------------------------------------------
async def vision_extract_pipeline(image_bytes: bytes) -> Dict[str, Any]:
    """
    1) OCR mode on full image
    2) If weak, micro-crop + preprocess + OCR (try a few)
    3) If still weak, visual tokens mode
    """
    # Step 1: OCR full
    ocr = openai_ocr_json(image_bytes)
    raw_text = (ocr.get("raw_text") or "").strip()
    visual_description = (ocr.get("visual_description") or "").strip()
    queries = [q for q in (ocr.get("queries") or []) if isinstance(q,str)]
    guesses = [g for g in (ocr.get("guesses") or []) if isinstance(g,str)]

    # Step 2: micro-crop OCR retries if text is empty/too short
    if len(raw_text) < 3:
        try:
            base = pil_from_bytes(image_bytes)
            for crop in micro_crops(base):
                pre = preprocess_ocr(crop)
                attempt = openai_ocr_json(bytes_from_pil(pre, fmt="JPEG"))
                if (attempt.get("raw_text") or "").strip():
                    # merge
                    raw_text = (raw_text + "\n" + attempt.get("raw_text","")).strip()
                    queries += attempt.get("queries") or []
                    guesses += attempt.get("guesses") or []
                    if not visual_description and attempt.get("visual_description"):
                        visual_description = attempt.get("visual_description")
                    if len(raw_text) > 3:
                        break
        except Exception:
            pass

    # Step 3: if still weak, go visual mode
    visual_json = {}
    if len(raw_text) < 3:
        visual_json = openai_visual_keywords_json(image_bytes)
        if not visual_description:
            visual_description = (visual_json.get("visual_description") or "").strip()

    return {
        "raw_text": raw_text,
        "visual_description": visual_description,
        "queries": queries,
        "guesses": guesses,
        "visual_json": visual_json
    }

async def search_discogs_with_queries(queries: List[str]) -> Tuple[Optional[str], List[Tuple[str,str]], List[str]]:
    used, all_candidates = [], []
    primary: Optional[str] = None

    for q in queries:
        used.append(q)
        results = await google_cse_search(q)
        discogs = filter_discogs(results)
        for url, thumb in discogs:
            all_candidates.append((url, thumb))
            if primary is None:
                primary = url
        if primary:  # stop early on first hit
            break

    # de-dup candidates
    seen=set(); uniq=[]
    for u,t in all_candidates:
        if u not in seen:
            uniq.append((u,t)); seen.add(u)
    return primary, uniq, used

# --- API Models --------------------------------------------------------------
class IdentifyResponse(BaseModel):
    discogs_url: Optional[str]
    confidence: Optional[float]
    alternates: List[str]
    used_queries: List[str]
    vision_text: str
    vision_description: str

# --- Route -------------------------------------------------------------------
@app.post("/identify", response_model=IdentifyResponse)
async def identify(file: UploadFile = File(...), rerank: bool = True):
    try:
        image_bytes = await file.read()

        # 1) Vision (OCR→micro-crop→visual)
        v = await vision_extract_pipeline(image_bytes)
        raw_text = v["raw_text"]
        visual_description = v["visual_description"]
        guesses = v["guesses"]
        queries_seed = v["queries"]

        # 2) Build Discogs queries (text-first, else visual)
        if raw_text.strip():
            used_queries = build_discogs_queries_from_text(raw_text, guesses, visual_description)
            # include any seeds from vision
            used_queries = (queries_seed or [])[:6] + used_queries
        else:
            used_queries = build_discogs_queries_from_visual(v["visual_json"])

        # 3) Google CSE → Discogs candidates
        discogs_url, candidates, used = await search_discogs_with_queries(used_queries)

        # 4) (Optional) visual re-rank on candidate thumbnails
        confidence = None
        alternates: List[str] = []
        if rerank and candidates:
            scored = []
            for url, thumb in candidates[:6]:
                s = await compare_visual_similarity(image_bytes, thumb)
                scored.append((s, url))
            scored.sort(reverse=True, key=lambda x: x[0])
            if scored:
                best_score, best_url = scored[0]
                discogs_url = best_url
                confidence = float(best_score)
                alternates = [u for _, u in scored[1:5]]
        else:
            alternates = [u for u,_ in candidates[1:6]]

        return JSONResponse({
            "discogs_url": discogs_url,
            "confidence": confidence if discogs_url else None,
            "alternates": alternates,
            "used_queries": used,
            "vision_text": raw_text,
            "vision_description": visual_description
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": "identify_failed",
                "message": str(e),
                "trace": traceback.format_exc()
            }
        )
