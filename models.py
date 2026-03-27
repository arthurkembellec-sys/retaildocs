from pydantic import BaseModel
from typing import Optional


class DocumentMetadata(BaseModel):
    retailer: Optional[str] = None
    country: Optional[str] = None
    airport: Optional[str] = None
    store: Optional[str] = None
    year: Optional[int] = None
    doc_type: Optional[str] = None


class SearchRequest(BaseModel):
    question: str
    retailer: Optional[str] = None
    country: Optional[str] = None
    airport: Optional[str] = None
    store: Optional[str] = None
    year: Optional[int] = None
    doc_type: Optional[str] = None


class ChunkResult(BaseModel):
    content: str
    similarity: float
    filename: str
    retailer: Optional[str]
    country: Optional[str]
    airport: Optional[str]
    store: Optional[str]
    year: Optional[int]
    doc_type: Optional[str]


class SearchResponse(BaseModel):
    answer: str
    sources: list[ChunkResult]
