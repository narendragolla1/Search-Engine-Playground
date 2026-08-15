from loguru import logger
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import uuid

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding

from qdrant_client import QdrantClient
from qdrant_client.http import models

@dataclass
class SearchResult:
    """Represents a single result from the search engine."""
    item: Dict[str, Any]
    score: float
    keyword_score: float = 0.0 # Keeping for compatibility
    semantic_score: float = 0.0 # Keeping for compatibility

class SearchEngine:
    """
    A generic hybrid search engine combining sparse (SPLADE) and dense (SentenceTransformers) search natively in Qdrant.
    """

    def __init__(self, model: SentenceTransformer, reranker: CrossEncoder = None):
        self.model = model
        self.reranker = reranker
        self.sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
        self.collection_name = "search_collection_v2" # Using a new collection for schema change
        
        # Initialize Qdrant Client (Using local memory/disk for now)
        self.qdrant = QdrantClient(path="./qdrant_data")
        
        self._init_collection()

    def _init_collection(self):
        """Initializes the Qdrant collection with dense and sparse vectors."""
        if not self.qdrant.collection_exists(self.collection_name):
            vector_size = self.model.get_sentence_embedding_dimension()
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "splade": models.SparseVectorParams()
                }
            )
            logger.info(f"Created Qdrant collection '{self.collection_name}' with dense (size {vector_size}) and sparse vectors.")

    def _build_searchable_text(self, item: Dict[str, Any], searchable_fields: List[str]) -> str:
        text_parts = []
        for field in searchable_fields:
            val = item.get(field)
            if val:
                text_parts.append(str(val))
        return " ".join(text_parts).lower()

    def _build_keyword_text(self, item: Dict[str, Any], searchable_fields: List[str], field_weights: Optional[Dict[str, float]] = None) -> str:
        """Builds text for SPLADE, repeating fields based on their weight for boosting."""
        text_parts = []
        field_weights = field_weights or {}
        
        for field in searchable_fields:
            val = item.get(field)
            if val:
                weight = int(field_weights.get(field, 1.0))
                weight = max(1, weight)
                for _ in range(weight):
                    text_parts.append(str(val))
                    
        return " ".join(text_parts).lower()

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
        if not data:
            logger.warning("Attempted to index empty data.")
            return

        logger.info(f"Indexing {len(data)} items into Qdrant...")
        
        texts_to_embed = []
        keyword_texts_to_embed = []
        point_ids = []
        payloads = []
        
        for item in data:
            point_ids.append(self._parse_point_id(item.get("id")))
                
            combined_text = self._build_searchable_text(item, searchable_fields)
            keyword_text = self._build_keyword_text(item, searchable_fields, field_weights)
            
            texts_to_embed.append(combined_text)
            keyword_texts_to_embed.append(keyword_text)
            
            payload = item.copy()
            payload["_searchable_text"] = combined_text
            payload["_searchable_fields"] = searchable_fields
            payloads.append(payload)

        logger.info("Generating dense embeddings...")
        dense_embeddings = self.model.encode(texts_to_embed, convert_to_numpy=True)
        
        logger.info("Generating sparse embeddings (SPLADE)...")
        sparse_embeddings = list(self.sparse_model.embed(keyword_texts_to_embed))
        
        points = []
        for i in range(len(data)):
            points.append(
                models.PointStruct(
                    id=point_ids[i],
                    vector={
                        "dense": dense_embeddings[i].tolist(),
                        "splade": models.SparseVector(
                            indices=sparse_embeddings[i].indices.tolist(),
                            values=sparse_embeddings[i].values.tolist()
                        )
                    },
                    payload=payloads[i]
                )
            )
            
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"Successfully indexed {len(points)} items.")

    def add_documents(self, documents: List[Dict[str, Any]], searchable_fields: List[str], field_weights: Optional[Dict[str, float]] = None):
        self.index(documents, searchable_fields, field_weights)
        
    def update_document(self, document_id: str, document: Dict[str, Any], searchable_fields: List[str], field_weights: Optional[Dict[str, float]] = None):
        doc = document.copy()
        doc["id"] = document_id
        self.index([doc], searchable_fields, field_weights)

    def delete_document(self, document_id: str):
        self.qdrant.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=[self._parse_point_id(document_id)],
            ),
        )

    def _build_qdrant_filter(self, filters: Dict[str, Any]) -> Optional[models.Filter]:
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
            else:
                must_conditions.append(
                    models.FieldCondition(
                        key=field,
                        match=models.MatchValue(value=filter_val)
                    )
                )
        
        if must_conditions:
            return models.Filter(must=must_conditions)
        return None

    def search(self, query: str, filters: Dict[str, Any] = None, top_k: int = 10, offset: int = 0) -> List[SearchResult]:
        if not self.qdrant.collection_exists(self.collection_name):
            logger.error("Search engine collection does not exist.")
            return []

        qdrant_filter = self._build_qdrant_filter(filters)
        fetch_limit = top_k + offset

        if not query.strip():
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
                    score=1.0
                ) for p in points
            ]

        query_lower = query.lower()
        query_dense = self.model.encode(query_lower, convert_to_numpy=True)
        query_sparse = list(self.sparse_model.query_embed(query_lower))[0]

        semantic_query = models.Prefetch(
            query=query_dense.tolist(),
            using="dense",
            limit=fetch_limit,
            filter=qdrant_filter
        )
        
        sparse_query = models.Prefetch(
            query=models.SparseVector(
                indices=query_sparse.indices.tolist(),
                values=query_sparse.values.tolist(),
            ),
            using="splade",
            limit=fetch_limit,
            filter=qdrant_filter
        )

        hybrid_results = self.qdrant.query_points(
            collection_name=self.collection_name,
            prefetch=[semantic_query, sparse_query],
            query=models.RrfQuery(
                rrf=models.Rrf()
            ),
            limit=top_k,
            offset=offset,
            with_payload=True,
        ).points

        results = []
        valid_pairs = []
        for point in hybrid_results:
            clean_item = {k:v for k,v in point.payload.items() if not k.startswith('_')}
            results.append(SearchResult(item=clean_item, score=point.score))
            valid_pairs.append((query_lower, point.payload.get("_searchable_text", "")))

        if self.reranker and valid_pairs and query.strip():
            rerank_scores = self.reranker.predict(valid_pairs)
            for i, r in enumerate(results):
                r.score = float(rerank_scores[i])
            results.sort(key=lambda x: x.score, reverse=True)

        return results
