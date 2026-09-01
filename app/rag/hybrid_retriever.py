"""
Hybrid Retrieval Module
Combines:
1. Dense Vector Search (pgvector cosine similarity)
2. Sparse Keyword Search (PostgreSQL full-text search tsvector)
3. Reciprocal Rank Fusion (RRF, k=60)
4. FlashRank Cross-Encoder Re-ranking
"""

from typing import List, Dict, Any, Optional
from sqlalchemy import text
from app.database import AsyncSessionLocal
from app.rag.ingest import get_embedding_model
from app.rag.reranker import rerank_passages


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]], 
    sparse_results: List[Dict[str, Any]], 
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Blends dense and sparse search rankings using Reciprocal Rank Fusion (RRF).
    Score = sum(1 / (k + rank))
    """
    scores: Dict[int, float] = {}
    doc_map: Dict[int, Dict[str, Any]] = {}

    # Rank dense results
    for rank, doc in enumerate(dense_results):
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

    # Rank sparse results
    for rank, doc in enumerate(sparse_results):
        doc_id = doc["id"]
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

    # Sort documents by fused RRF score descending
    sorted_doc_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    fused_docs = []
    for did in sorted_doc_ids:
        doc = doc_map[did]
        doc["rrf_score"] = scores[did]
        fused_docs.append(doc)
    return fused_docs


async def hybrid_search_raw(
    query_text: str, 
    query_embedding: List[float], 
    candidate_fetch_limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Executes dense cosine similarity search and PostgreSQL BM25 FTS in parallel,
    then combines them via RRF.
    """
    async with AsyncSessionLocal() as session:
        # 1. Dense Vector Search (pgvector cosine distance)
        dense_query = text("""
            SELECT id, document_id, document_name, chunk_index, content, metadata_json,
                   1 - (embedding <=> CAST(:vector AS vector)) AS similarity_score
            FROM document_chunks
            ORDER BY embedding <=> CAST(:vector AS vector)
            LIMIT :fetch_limit;
        """)
        
        # 2. Sparse BM25 / Full-Text Search (PostgreSQL tsvector)
        sparse_query = text("""
            SELECT id, document_id, document_name, chunk_index, content, metadata_json,
                   ts_rank_cd(tsv_content, plainto_tsquery('english', :query)) AS bm25_score
            FROM document_chunks
            WHERE tsv_content @@ plainto_tsquery('english', :query)
            ORDER BY bm25_score DESC
            LIMIT :fetch_limit;
        """)
        
        dense_res = await session.execute(
            dense_query, 
            {"vector": str(query_embedding), "fetch_limit": candidate_fetch_limit}
        )
        sparse_res = await session.execute(
            sparse_query, 
            {"query": query_text, "fetch_limit": candidate_fetch_limit}
        )
        
        dense_docs = [dict(row._mapping) for row in dense_res]
        sparse_docs = [dict(row._mapping) for row in sparse_res]
        
        # Blend candidates via Reciprocal Rank Fusion
        fused_docs = reciprocal_rank_fusion(dense_docs, sparse_docs)
        return fused_docs


async def retrieve_context(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    End-to-End Retrieval Pipeline:
    1. Validates query input.
    2. Generates 384-d dense embedding of query.
    3. Runs hybrid search (Dense + BM25) to fetch top candidate chunks.
    4. Cross-Encoder re-ranks candidate chunks.
    5. Returns top_k highest relevance chunks.
    """
    clean_query = query.strip() if query else ""
    if not clean_query:
        return []

    # 1. Embed query
    model = get_embedding_model()
    query_vector = model.encode(clean_query, normalize_embeddings=True).tolist()

    # 2. Hybrid search with oversampling (fetch 2x candidates)
    candidate_limit = max(top_k * 3, 10)
    fused_candidates = await hybrid_search_raw(
        query_text=clean_query, 
        query_embedding=query_vector, 
        candidate_fetch_limit=candidate_limit
    )

    # 3. FlashRank Cross-Encoder Re-ranking
    final_ranked_chunks = rerank_passages(clean_query, fused_candidates, top_n=top_k)
    return final_ranked_chunks
