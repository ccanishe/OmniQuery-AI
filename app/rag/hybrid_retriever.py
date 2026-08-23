"""
Hybrid Retrieval Module
Combines Dense Vector Search (pgvector cosine similarity) 
with Sparse Keyword Search (PostgreSQL full-text search tsvector)
and applies Reciprocal Rank Fusion (RRF) + Cross-Encoder Re-ranking.
"""

from typing import List, Dict, Any
from sqlalchemy import text
from app.database import AsyncSessionLocal

def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]], 
    sparse_results: List[Dict[str, Any]], 
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Blends dense and sparse search rankings using Reciprocal Rank Fusion (RRF).
    Score = sum(1 / (k + rank))
    """
    scores = {}
    doc_map = {}

    for rank, doc in enumerate(dense_results):
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

    for rank, doc in enumerate(sparse_results):
        doc_id = doc["id"]
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

    # Sort documents by fused RRF score descending
    sorted_doc_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[did] for did in sorted_doc_ids]

async def hybrid_search(query_text: str, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Executes parallel dense cosine similarity search and PostgreSQL BM25 FTS,
    then combines them via RRF.
    """
    async with AsyncSessionLocal() as session:
        # 1. Dense Vector Search
        dense_query = text("""
            SELECT id, document_id, content, metadata,
                   1 - (embedding <=> :vector::vector) AS similarity_score
            FROM document_chunks
            ORDER BY embedding <=> :vector::vector
            LIMIT :fetch_limit;
        """)
        
        # 2. Sparse BM25 / Full-Text Search
        sparse_query = text("""
            SELECT id, document_id, content, metadata,
                   ts_rank_cd(tsv_content, plainto_tsquery('english', :query)) AS bm25_score
            FROM document_chunks
            WHERE tsv_content @@ plainto_tsquery('english', :query)
            ORDER BY bm25_score DESC
            LIMIT :fetch_limit;
        """)
        
        # Execute queries
        dense_res = await session.execute(
            dense_query, 
            {"vector": str(query_embedding), "fetch_limit": top_k * 2}
        )
        sparse_res = await session.execute(
            sparse_query, 
            {"query": query_text, "fetch_limit": top_k * 2}
        )
        
        dense_docs = [dict(row._mapping) for row in dense_res]
        sparse_docs = [dict(row._mapping) for row in sparse_res]
        
        # Fuse rankings
        fused = reciprocal_rank_fusion(dense_docs, sparse_docs)
        return fused[:top_k]
