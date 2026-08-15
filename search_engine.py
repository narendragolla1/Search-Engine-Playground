import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a single result from the search engine."""
    item: Dict[str, Any]
    score: float
    keyword_score: float
    semantic_score: float

class SearchEngine:
    """
    A generic hybrid search engine combining keyword (BM25) and semantic (SentenceTransformers) search.
    """

    def __init__(self, model: SentenceTransformer):
        """
        Initializes the search engine.
        
        Args:
            model (SentenceTransformer): The pre-loaded SentenceTransformer model to use for semantic search.
        """
        self.model = model

        self.data: List[Dict[str, Any]] = []
        self.corpus_text: List[str] = []
        self.bm25: Optional[BM25Okapi] = None
        self.embeddings: Optional[np.ndarray] = None

    def index(self, data: List[Dict[str, Any]], searchable_fields: List[str]) -> None:
        """
        Indexes the provided data for both keyword and semantic search.

        Args:
            data (List[Dict[str, Any]]): The list of JSON-like objects to index.
            searchable_fields (List[str]): The keys in the dictionaries to use for searching.
        """
        if not data:
            logger.warning("Attempted to index empty data.")
            return

        self.data = data
        self.corpus_text = []

        # Prepare the text corpus by concatenating searchable fields
        for item in self.data:
            text_parts = []
            for field in searchable_fields:
                val = item.get(field)
                if val:
                    text_parts.append(str(val))
            combined_text = " ".join(text_parts).lower()
            self.corpus_text.append(combined_text)

        if not self.corpus_text:
            logger.warning("No searchable text found in the provided fields.")
            return

        # 1. Build Keyword Index (BM25)
        logger.info("Building BM25 keyword index...")
        tokenized_corpus = [doc.split() for doc in self.corpus_text]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 2. Build Semantic Index (Embeddings)
        logger.info("Generating semantic embeddings... This may take a while depending on data size.")
        self.embeddings = self.model.encode(self.corpus_text, convert_to_numpy=True)
        
        logger.info(f"Successfully indexed {len(self.data)} items.")

    def _passes_filters(self, item: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Helper to check if an item satisfies all filters."""
        if not filters:
            return True
            
        for field, filter_val in filters.items():
            item_val = item.get(field)
            if item_val is None:
                return False
                
            # Range filter (Numeric)
            if isinstance(filter_val, dict) and 'min' in filter_val and 'max' in filter_val:
                try:
                    # Attempt to parse as float (handle strings like "9.3" or "2,559")
                    numeric_val = float(str(item_val).replace(',', ''))
                    if not (filter_val['min'] <= numeric_val <= filter_val['max']):
                        return False
                except ValueError:
                    return False
            
            # Array filter (Categorical)
            elif isinstance(filter_val, list):
                if not filter_val:
                    continue # Empty selection means no restriction
                
                if isinstance(item_val, str):
                    # Check if string contains any of the selected categories
                    match = any(cat.lower() in item_val.lower() for cat in filter_val)
                    if not match:
                        return False
                elif isinstance(item_val, list):
                    match = any(cat in item_val for cat in filter_val)
                    if not match:
                        return False
                else:
                    if item_val not in filter_val:
                        return False
        return True

    def search(self, query: str, filters: Dict[str, Any] = None, alpha: float = 0.5, top_k: int = 10) -> List[SearchResult]:
        """
        Performs a hybrid search with optional pre-filtering.

        Args:
            query (str): The search query.
            filters (Dict[str, Any]): Filters to apply. Example: {"Rating": {"min": 8, "max": 10}, "Genre": ["Action"]}
            alpha (float): The weight of the keyword score (0.0 to 1.0). 
                           0.0 means semantic only, 1.0 means keyword only.
            top_k (int): The maximum number of results to return.

        Returns:
            List[SearchResult]: The top k search results sorted by the combined score.
        """
        if not self.data or self.bm25 is None or self.embeddings is None:
            logger.error("Search engine is not indexed. Call `index()` first.")
            return []

        filters = filters or {}
        
        # 1. Pre-filter items
        valid_indices = []
        for i, item in enumerate(self.data):
            if self._passes_filters(item, filters):
                valid_indices.append(i)
                
        if not valid_indices:
            return []

        if not query.strip():
            # If no text query, just return the filtered items
            results = []
            for idx in valid_indices[:top_k]:
                results.append(
                    SearchResult(
                        item=self.data[idx],
                        score=1.0,
                        keyword_score=0.0,
                        semantic_score=0.0
                    )
                )
            return results

        query_lower = query.lower()

        # 2. Keyword Scores
        tokenized_query = query_lower.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize BM25 scores to [0, 1] for fair comparison with cosine similarity
        if np.max(bm25_scores) > 0:
            bm25_scores = bm25_scores / np.max(bm25_scores)

        # 3. Semantic Scores
        query_embedding = self.model.encode(query_lower, convert_to_numpy=True)
        semantic_scores = cos_sim(query_embedding, self.embeddings)[0].numpy()
        semantic_scores = np.clip(semantic_scores, 0, 1)

        # 4. Hybrid Scoring
        combined_scores = (alpha * bm25_scores) + ((1.0 - alpha) * semantic_scores)

        # Mask out items that didn't pass the filters
        mask = np.zeros(len(self.data), dtype=bool)
        mask[valid_indices] = True
        combined_scores[~mask] = -1.0

        # 5. Rank and format results
        top_indices = np.argsort(combined_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            # Only include results with a score > 0 and that passed the filter mask
            if combined_scores[idx] > 0 and mask[idx]:
                results.append(
                    SearchResult(
                        item=self.data[idx],
                        score=float(combined_scores[idx]),
                        keyword_score=float(bm25_scores[idx]),
                        semantic_score=float(semantic_scores[idx])
                    )
                )

        return results
