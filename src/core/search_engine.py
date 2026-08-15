from loguru import logger
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import uuid

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import cos_sim

from qdrant_client import QdrantClient
from qdrant_client.http import models

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
    Data and dense embeddings are stored in Qdrant.
    """

    def __init__(self, model: SentenceTransformer, reranker: CrossEncoder = None):
        """
        Initializes the search engine.
        
        Args:
            model (SentenceTransformer): The pre-loaded SentenceTransformer model to use for semantic search.
            reranker (CrossEncoder, optional): The cross-encoder model for Stage 2 re-ranking.
        """
        self.model = model
        self.reranker = reranker
        self.collection_name = "search_collection"
        
        # Initialize Qdrant Client (Using local memory/disk for now, can be swapped for Qdrant Cloud)
        self.qdrant = QdrantClient(path="./qdrant_data")
        
        # We will keep BM25 in memory for Phase 1. It will be replaced by SPLADE in Qdrant in Phase 3.
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_corpus_map: Dict[str, str] = {} # Map ID to text for BM25
        
        self._init_collection()

    def _init_collection(self):
        """Initializes the Qdrant collection if it doesn't exist."""
        if not self.qdrant.collection_exists(self.collection_name):
            vector_size = self.model.get_sentence_embedding_dimension()
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created Qdrant collection '{self.collection_name}' with vector size {vector_size}.")

    def _build_searchable_text(self, item: Dict[str, Any], searchable_fields: List[str]) -> str:
        text_parts = []
        for field in searchable_fields:
            val = item.get(field)
            if val:
                text_parts.append(str(val))
        return " ".join(text_parts).lower()

    def _build_keyword_text(self, item: Dict[str, Any], searchable_fields: List[str], field_weights: Optional[Dict[str, float]] = None) -> str:
        """Builds text for BM25, repeating fields based on their weight for boosting."""
        text_parts = []
        field_weights = field_weights or {}
        
        for field in searchable_fields:
            val = item.get(field)
            if val:
                # Default weight is 1.0. If weight is 3.0, we repeat the text 3 times.
                weight = int(field_weights.get(field, 1.0))
                # Ensure at least 1 repetition
                weight = max(1, weight)
                for _ in range(weight):
                    text_parts.append(str(val))
                    
        return " ".join(text_parts).lower()

    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """Tokenizes text into words and n-grams for typo tolerance."""
        words = text.split()
        tokens = []
        for word in words:
            tokens.append(word)
            # Generate 3-grams for words longer than 3 characters
            if len(word) > 3:
                for i in range(len(word) - 2):
                    tokens.append(word[i:i+3])
        return tokens

    def _rebuild_bm25(self):
        """Rebuilds the in-memory BM25 index from the current qdrant payloads."""
        logger.info("Rebuilding in-memory BM25 index with N-grams...")
        scroll_res = self.qdrant.scroll(
            collection_name=self.collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
        points, _ = scroll_res
        
        self.bm25_corpus_map = {}
        corpus_texts = []
        
        for point in points:
            text = point.payload.get("_keyword_text", point.payload.get("_searchable_text", ""))
            self.bm25_corpus_map[point.id] = text
            corpus_texts.append(self._tokenize_for_bm25(text))
            
        if corpus_texts:
            self.bm25 = BM25Okapi(corpus_texts)
        else:
            self.bm25 = None
            
        # Store a mapping of internal BM25 integer index to Qdrant point ID
        self.bm25_id_map = list(self.bm25_corpus_map.keys())

    def _parse_point_id(self, raw_id: Any) -> Any:
        if raw_id is None:
            return str(uuid.uuid4())
        try:
            return int(raw_id)
        except ValueError:
            try:
                uuid.UUID(str(raw_id))
                return str(raw_id)
            except ValueError:
                return str(uuid.uuid5(uuid.NAMESPACE_OID, str(raw_id)))

    def index(self, data: List[Dict[str, Any]], searchable_fields: List[str], field_weights: Optional[Dict[str, float]] = None) -> None:
        """
        Indexes the provided data by inserting it into Qdrant.
        """
        if not data:
            logger.warning("Attempted to index empty data.")
            return

        logger.info(f"Indexing {len(data)} items into Qdrant...")
        
        points = []
        texts_to_embed = []
        
        for item in data:
            point_id = self._parse_point_id(item.get("id"))
                
            combined_text = self._build_searchable_text(item, searchable_fields)
            keyword_text = self._build_keyword_text(item, searchable_fields, field_weights)
            texts_to_embed.append(combined_text)
            
            # Store the searchable text in the payload for BM25 and Reranker
            payload = item.copy()
            payload["_searchable_text"] = combined_text
            payload["_keyword_text"] = keyword_text
            payload["_searchable_fields"] = searchable_fields
            
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=[], # Will fill next
                    payload=payload
                )
            )

        # Generate dense embeddings
        logger.info("Generating dense embeddings...")
        embeddings = self.model.encode(texts_to_embed, convert_to_numpy=True)
        
        for i, point in enumerate(points):
            point.vector = embeddings[i].tolist()
            
        # Upsert to Qdrant
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        self._rebuild_bm25()
        logger.info(f"Successfully indexed {len(points)} items.")

    def add_documents(self, documents: List[Dict[str, Any]], searchable_fields: List[str], field_weights: Optional[Dict[str, float]] = None):
        """Adds new documents to the index."""
        self.index(documents, searchable_fields, field_weights)
        
    def update_document(self, document_id: str, document: Dict[str, Any], searchable_fields: List[str], field_weights: Optional[Dict[str, float]] = None):
        """Updates an existing document."""
        doc = document.copy()
        doc["id"] = document_id
        self.index([doc], searchable_fields, field_weights)

    def delete_document(self, document_id: str):
        """Deletes a document from the index by ID."""
        self.qdrant.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=[self._parse_point_id(document_id)],
            ),
        )
        self._rebuild_bm25()

    def _build_qdrant_filter(self, filters: Dict[str, Any]) -> Optional[models.Filter]:
        """Converts user filters into Qdrant Filter models."""
        if not filters:
            return None
            
        must_conditions = []
        for field, filter_val in filters.items():
            if isinstance(filter_val, dict) and 'min' in filter_val and 'max' in filter_val:
                must_conditions.append(
                    models.FieldCondition(
                        key=field,
                        range=models.Range(
                            gte=filter_val['min'],
                            lte=filter_val['max']
                        )
                    )
                )
            elif isinstance(filter_val, list) and filter_val:
                must_conditions.append(
                    models.FieldCondition(
                        key=field,
                        match=models.MatchAny(any=filter_val)
                    )
                )
        
        if must_conditions:
            return models.Filter(must=must_conditions)
        return None

    def search(self, query: str, filters: Dict[str, Any] = None, top_k: int = 10, offset: int = 0, rrf_k: int = 60) -> List[SearchResult]:
        """
        Performs a hybrid search using Qdrant for semantic + filtering, and in-memory BM25 for keyword.
        Supports offset-based pagination.
        """
        if not self.qdrant.collection_exists(self.collection_name):
            logger.error("Search engine collection does not exist.")
            return []

        qdrant_filter = self._build_qdrant_filter(filters)
        
        # We need to fetch enough candidates to cover the offset + top_k
        fetch_limit = top_k + offset

        if not query.strip():
            # Filter only
            scroll_res = self.qdrant.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_filter,
                offset=offset,
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )
            points, _ = scroll_res
            return [
                SearchResult(
                    item={k:v for k,v in p.payload.items() if not k.startswith('_')},
                    score=1.0, keyword_score=0.0, semantic_score=0.0
                ) for p in points
            ]

        query_lower = query.lower()
        
        # 1. Semantic Search (Qdrant)
        query_embedding = self.model.encode(query_lower, convert_to_numpy=True)
        semantic_results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            query_filter=qdrant_filter,
            limit=fetch_limit * 2, # Fetch more for RRF
            with_payload=True
        ).points
        
        semantic_ranks = {res.id: rank + 1 for rank, res in enumerate(semantic_results)}
        semantic_scores = {res.id: res.score for res in semantic_results}

        # 2. Keyword Search (BM25)
        bm25_ranks = {}
        bm25_scores = {}
        if self.bm25:
            tokenized_query = self._tokenize_for_bm25(query_lower)
            all_bm25_scores = self.bm25.get_scores(tokenized_query)
            
            sorted_indices = np.argsort(all_bm25_scores)[::-1]
            
            valid_qdrant_ids = set()
            if qdrant_filter:
                scroll_res = self.qdrant.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=qdrant_filter,
                    limit=10000,
                    with_payload=False
                )
                valid_qdrant_ids = {p.id for p in scroll_res[0]}
            else:
                valid_qdrant_ids = set(self.bm25_id_map)
                
            current_rank = 1
            for idx in sorted_indices:
                point_id = self.bm25_id_map[idx]
                score = all_bm25_scores[idx]
                if score > 0 and point_id in valid_qdrant_ids:
                    bm25_ranks[point_id] = current_rank
                    bm25_scores[point_id] = score
                    current_rank += 1

        # 3. RRF Scoring
        all_candidate_ids = set(semantic_ranks.keys()).union(set(bm25_ranks.keys()))
        rrf_scores = {}
        
        for point_id in all_candidate_ids:
            score = 0.0
            if point_id in semantic_ranks:
                score += 1.0 / (rrf_k + semantic_ranks[point_id])
            if point_id in bm25_ranks:
                score += 1.0 / (rrf_k + bm25_ranks[point_id])
            rrf_scores[point_id] = score
            
        # Sort by RRF and apply pagination offset and limit
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        paginated_ids = sorted_ids[offset : offset + top_k]
        
        # 4. Fetch final payloads from Qdrant
        if not paginated_ids:
            return []
            
        final_points = self.qdrant.retrieve(
            collection_name=self.collection_name,
            ids=paginated_ids,
            with_payload=True
        )
        
        points_map = {p.id: p for p in final_points}
        
        results = []
        valid_pairs = []
        
        for point_id in paginated_ids:
            if point_id not in points_map:
                continue
            point = points_map[point_id]
            clean_item = {k:v for k,v in point.payload.items() if not k.startswith('_')}
            results.append(
                SearchResult(
                    item=clean_item,
                    score=rrf_scores[point_id],
                    keyword_score=bm25_scores.get(point_id, 0.0),
                    semantic_score=semantic_scores.get(point_id, 0.0)
                )
            )
            valid_pairs.append((query_lower, point.payload.get("_searchable_text", "")))

        # 5. Cross-Encoder Re-Ranking (Stage 2)
        if self.reranker and valid_pairs and query.strip():
            rerank_scores = self.reranker.predict(valid_pairs)
            for i, r in enumerate(results):
                r.score = float(rerank_scores[i])
            results.sort(key=lambda x: x.score, reverse=True)

        return results
