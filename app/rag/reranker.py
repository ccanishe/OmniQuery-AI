"""
FlashRank Cross-Encoder Re-ranker Module
Applies deep cross-attention over fused (Dense + BM25) candidate chunks
to compute joint query-passage relevance scores and eliminate context dilution.
"""

from typing import List, Dict, Any, Optional
import os

try:
    from flashrank import Ranker, RerankRequest
    FLASHRANK_AVAILABLE = True
except ImportError:
    FLASHRANK_AVAILABLE = False

_ranker_instance = None


def get_ranker() -> Optional[Any]:
    """Lazy-loads and caches the FlashRank cross-encoder model."""
    global _ranker_instance
    if not FLASHRANK_AVAILABLE:
        return None
    if _ranker_instance is None:
        try:
            print("Initializing FlashRank cross-encoder model (ms-marco-TinyBERT-L-2-v2)...")
            _ranker_instance = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/tmp/flashrank")
        except Exception as e:
            print(f"Warning: FlashRank initialization failed: {e}")
            _ranker_instance = None
    return _ranker_instance


def rerank_passages(
    query: str, 
    candidates: List[Dict[str, Any]], 
    top_n: int = 3
) -> List[Dict[str, Any]]:
    """
    Takes top fused candidates and reranks them using joint cross-attention.
    Returns the top_n most contextually precise chunks.
    """
    if not candidates:
        return []

    ranker = get_ranker()
    if ranker is None:
        # Fallback to top fused results if ranker unavailable
        return candidates[:top_n]

    # Format passages for FlashRank safely
    passages = []
    for idx, doc in enumerate(candidates):
        raw_content = doc.get("content")
        text_content = str(raw_content).strip() if raw_content is not None else ""
        passages.append({
            "id": doc.get("id", idx),
            "text": text_content,
            "meta": doc
        })

    try:
        rerank_req = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(rerank_req)
        
        # Extract and format top reranked items
        reranked_docs = []
        for item in results[:top_n]:
            doc_data = item.get("meta", {})
            doc_data["rerank_score"] = item.get("score", 0.0)
            reranked_docs.append(doc_data)
        return reranked_docs
    except Exception as e:
        print(f"FlashRank reranking error ({e}), falling back to RRF order.")
        return candidates[:top_n]
