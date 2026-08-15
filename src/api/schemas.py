from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class IndexRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="The list of JSON objects to index")
    searchable_fields: List[str] = Field(..., description="The fields in the JSON objects to build the search index upon")

class IndexResponse(BaseModel):
    message: str
    items_indexed: int

class SearchRequest(BaseModel):
    query: str = Field(default="", description="The search query string")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional filters to apply before searching")
    top_k: int = Field(default=10, description="The number of results to return")

class SearchResultItem(BaseModel):
    item: Dict[str, Any]
    score: float
    keyword_score: float
    semantic_score: float

class SearchResponse(BaseModel):
    results: List[SearchResultItem]
