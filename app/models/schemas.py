from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional

class VisionExtract(BaseModel):
    artist: Optional[str] = None
    title: Optional[str] = None
    label: Optional[str] = None
    catalog_no: Optional[str] = Field(None, alias="catalogNo")
    year: Optional[str] = None
    country: Optional[str] = None
    confidence_notes: Optional[str] = None
    raw_text: Optional[str] = None
    keywords: List[str] = []

class DiscogsResult(BaseModel):
    id: int
    type: str
    title: str
    year: Optional[int] = None
    country: Optional[str] = None
    label: Optional[List[str]] = None
    catno: Optional[str] = None
    resource_url: Optional[HttpUrl] = None
    uri: Optional[str] = None  # web URI path e.g. /release/1234

class IdentifyResponse(BaseModel):
    source_image: Optional[HttpUrl] = None
    vision: VisionExtract
    best_guess: Optional[DiscogsResult] = None
    candidates: List[DiscogsResult] = []
    notes: Optional[str] = None
