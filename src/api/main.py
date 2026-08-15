from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
from typing import Dict, Any

from src.api.schemas import (
    IndexRequest, IndexResponse, SearchRequest, SearchResponse, 
    SearchResultItem, ChatRequest, UpdateDocumentRequest, GenericResponse,
    SchemaAnalysisRequest, SchemaAnalysisResponse
)
from src.api.dependencies import init_search_engine, get_search_engine
from src.core.search_engine import SearchEngine
from src.api.llm_service import generate_chat_stream, analyze_schema, extract_intent

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML models exactly once
    init_search_engine()
    yield
    # Shutdown logic (if any)

app = FastAPI(title="Search Engine API", lifespan=lifespan)

# Allow CORS for local frontend (Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Advanced Search Engine API"}

@app.post("/schema/analyze", response_model=SchemaAnalysisResponse)
async def analyze_schema_endpoint(request: SchemaAnalysisRequest):
    try:
        result = await analyze_schema(request.sample_data)
        return SchemaAnalysisResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index", response_model=IndexResponse)
async def index_data(request: IndexRequest, engine: SearchEngine = Depends(get_search_engine)):
    try:
        await run_in_threadpool(engine.index, request.data, request.searchable_fields, request.field_weights)
        return IndexResponse(
            message="Successfully indexed data.",
            items_indexed=len(request.data)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/document", response_model=GenericResponse)
async def update_document(request: UpdateDocumentRequest, engine: SearchEngine = Depends(get_search_engine)):
    try:
        await run_in_threadpool(engine.update_document, request.document_id, request.document, request.searchable_fields, request.field_weights)
        return GenericResponse(message=f"Successfully updated document {request.document_id}.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/document/{document_id}", response_model=GenericResponse)
async def delete_document(document_id: str, engine: SearchEngine = Depends(get_search_engine)):
    try:
        await run_in_threadpool(engine.delete_document, document_id)
        return GenericResponse(message=f"Successfully deleted document {document_id}.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, engine: SearchEngine = Depends(get_search_engine)):
    try:
        filters = request.filters or {}
        
        if request.extract_intent and request.filterable_fields and request.query:
            extracted_filters = await extract_intent(request.query, request.filterable_fields)
            if extracted_filters:
                filters.update(extracted_filters)
                
        results = await run_in_threadpool(
            engine.search,
            query=request.query,
            filters=filters,
            top_k=request.top_k,
            offset=request.offset
        )
        
        # Convert internal SearchResult objects to Pydantic Response schemas
        formatted_results = [
            SearchResultItem(
                item=r.item,
                score=r.score,
                keyword_score=r.keyword_score,
                semantic_score=r.semantic_score
            ) for r in results
        ]
        
        return SearchResponse(results=formatted_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        stream = generate_chat_stream(request.messages, request.context)
        return StreamingResponse(stream, media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
