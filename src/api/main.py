from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.api.schemas import IndexRequest, IndexResponse, SearchRequest, SearchResponse, SearchResultItem
from src.api.dependencies import init_search_engine, get_search_engine
from src.core.search_engine import SearchEngine

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

@app.post("/index", response_model=IndexResponse)
def index_data(request: IndexRequest, engine: SearchEngine = Depends(get_search_engine)):
    try:
        engine.index(request.data, request.searchable_fields)
        return IndexResponse(
            message="Successfully indexed data.",
            items_indexed=len(request.data)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest, engine: SearchEngine = Depends(get_search_engine)):
    try:
        results = engine.search(
            query=request.query,
            filters=request.filters,
            top_k=request.top_k
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
