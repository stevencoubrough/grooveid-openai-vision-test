# app/models/schemas.py

from typing import Optional, List, Dict, Any
from pydantic import BaseModel

class DiscogsResult(BaseModel):
    id: Optional[int] = None
    type: Optional[str] = None
    title: Optional[str] = None
    year: Optional[int] = None
    country: Optional[str] = None
    label: Optional[List[str]] = None
    catno: Optional[str] = None
    resource_url: Optional[str] = None
    uri: Optional[str] = None

class IdentifyResponse(BaseModel):
    source_image: Optional[str] = None
    vision: Dict[str, Any]
    best_guess: Optional[DiscogsResult] = None
    candidates: List[DiscogsResult] = []
    notes: Optional[str] = None
