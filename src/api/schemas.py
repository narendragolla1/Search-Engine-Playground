from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class IndexRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="The list of JSON objects to index")
    searchable_fields: List[str] = Field(..., description="The fields in the JSON objects to build the search index upon")
    field_weights: Optional[Dict[str, float]] = Field(default=None, description="Optional weights for each field. e.g. {'Title': 2.0}")
    background: bool = Field(default=False, description="If true, indexing will happen in the background and return immediately")

class IndexResponse(BaseModel):
    message: str
    items_indexed: int
    task_id: Optional[str] = None

class UpdateDocumentRequest(BaseModel):
    document_id: str
    document: Dict[str, Any]
    searchable_fields: List[str]
    field_weights: Optional[Dict[str, float]] = Field(default=None, description="Optional weights for each field.")

class GenericResponse(BaseModel):
    message: str

class SchemaAnalysisRequest(BaseModel):
    sample_data: List[Dict[str, Any]] = Field(..., description="Sample JSON data to analyze")

class SchemaAnalysisResponse(BaseModel):
    searchable_fields: List[str]
    field_weights: Dict[str, float]
    filterable_fields: List[str]

class SearchRequest(BaseModel):
    query: str = Field(default="", description="The search query string")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional filters to apply before searching")
    top_k: int = Field(default=10, description="The number of results to return")
    offset: int = Field(default=0, description="Offset for pagination")
    extract_intent: bool = Field(default=False, description="If true, use LLM to extract filters from query")
    filterable_fields: Optional[List[str]] = Field(default=None, description="List of valid filter fields for intent extraction")

class SearchResultItem(BaseModel):
    item: Dict[str, Any]
    score: float

class SearchResponse(BaseModel):
    results: List[SearchResultItem]
