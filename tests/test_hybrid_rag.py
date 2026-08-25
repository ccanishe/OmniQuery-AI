"""
Unit Tests for OmniQuery-AI Week 1 Components:
1. Recursive Text Chunking
2. Reciprocal Rank Fusion (RRF) Ranking Logic
3. FlashRank Re-ranker Fallback
4. LangGraph Intent Classification Router
"""

import pytest
from app.rag.ingest import chunk_documents
from app.rag.hybrid_retriever import reciprocal_rank_fusion
from app.rag.reranker import rerank_passages
from app.agents.router import classify_intent_node


def test_chunk_documents():
    """Verifies that large texts are split into overlapping chunks with metadata."""
    sample_docs = [{
        "content": "A" * 1200,
        "metadata": {
            "document_name": "TestDoc.pdf",
            "document_id": "test_doc"
        }
    }]
    
    chunks = chunk_documents(sample_docs, chunk_size=500, chunk_overlap=50)
    assert len(chunks) >= 2
    assert chunks[0]["document_id"] == "test_doc"
    assert chunks[0]["chunk_index"] == 1
    assert "char_count" in chunks[0]["metadata"]


def test_reciprocal_rank_fusion():
    """Verifies that RRF accurately combines dense and sparse rankings."""
    dense_results = [
        {"id": 1, "content": "Doc A (Dense #1)"},
        {"id": 2, "content": "Doc B (Dense #2)"},
        {"id": 3, "content": "Doc C (Dense #3)"},
    ]
    
    sparse_results = [
        {"id": 2, "content": "Doc B (Sparse #1)"},
        {"id": 4, "content": "Doc D (Sparse #2)"},
        {"id": 1, "content": "Doc A (Sparse #3)"},
    ]
    
    # Doc 1 ranks (Dense: 1, Sparse: 3) -> 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226
    # Doc 2 ranks (Dense: 2, Sparse: 1) -> 1/(60+2) + 1/(60+1) = 0.01612 + 0.01639 = 0.03251
    # Therefore, Doc 2 should be ranked #1 overall
    fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60)
    
    assert len(fused) == 4
    assert fused[0]["id"] == 2
    assert fused[1]["id"] == 1
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]


def test_reranker_passages():
    """Verifies reranking output format and resilience."""
    candidates = [
        {"id": 1, "content": "Company refund policy within 30 days."},
        {"id": 2, "content": "Server error 500 troubleshooting guide."}
    ]
    results = rerank_passages("How do I get a refund?", candidates, top_n=1)
    assert len(results) == 1
    assert results[0]["id"] in [1, 2]


def test_langgraph_intent_router():
    """Verifies multi-agent intent routing classifications."""
    # Test SQL routing
    sql_state = classify_intent_node({"query": "How many total orders were placed last month?", "query_type": None})
    assert sql_state["query_type"] == "sql"

    # Test RAG routing
    rag_state = classify_intent_node({"query": "What is the company remote work policy on internet reimbursement?", "query_type": None})
    assert rag_state["query_type"] == "rag"

    # Test Direct conversation routing
    direct_state = classify_intent_node({"query": "Hello, who are you?", "query_type": None})
    assert direct_state["query_type"] == "direct"
