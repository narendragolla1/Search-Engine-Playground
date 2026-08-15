from sentence_transformers import SentenceTransformer, CrossEncoder
from src.core.search_engine import SearchEngine
import logging

logger = logging.getLogger(__name__)

class AppState:
    engine: SearchEngine = None

app_state = AppState()

def init_search_engine():
    """Initializes the ML models and Search Engine."""
    logger.info("Loading NLP models for Search Engine...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    app_state.engine = SearchEngine(model=model, reranker=reranker)
    logger.info("Search Engine initialized successfully.")

def get_search_engine() -> SearchEngine:
    """Dependency injection provider for the SearchEngine."""
    if app_state.engine is None:
        raise RuntimeError("Search Engine is not initialized")
    return app_state.engine
