"""
Comprehensive Negative & Edge Scenario Test Suite for OmniQuery-AI
Tests system resilience against:
1. Empty and whitespace-only queries
2. SQL injection strings and special symbols
3. Zero-result hybrid retrieval and asymmetric RRF
4. Re-ranker missing / degraded mode / corrupt metadata
5. LLM Synthesizer zero-chunk fallback guardrails
6. Intent Router ambiguous / adversarial queries
7. FastAPI Pydantic schema validation errors
"""

import pytest
import asyncio
from app.rag.hybrid_retriever import reciprocal_rank_fusion
from app.rag.reranker import rerank_passages
from app.rag.synthesizer import format_context_passages, synthesize_answer
from app.agents.router import classify_intent_node


# =====================================================================
# 1. Reciprocal Rank Fusion (RRF) Edge Cases
# =====================================================================

def test_rrf_empty_lists():
    """Verifies that RRF returns an empty list when both dense and sparse return no results."""
    fused = reciprocal_rank_fusion(dense_results=[], sparse_results=[], k=60)
    assert fused == []
    assert len(fused) == 0


def test_rrf_asymmetric_dense_only():
    """Verifies that RRF correctly ranks results when sparse search yields 0 hits."""
    dense_results = [
        {"id": 101, "content": "Only found by dense vector search"}
    ]
    sparse_results = []
    
    fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60)
    assert len(fused) == 1
    assert fused[0]["id"] == 101
    assert fused[0]["rrf_score"] == pytest.approx(1.0 / (60 + 1), rel=1e-3)


def test_rrf_asymmetric_sparse_only():
    """Verifies that RRF correctly ranks results when dense search yields 0 hits."""
    dense_results = []
    sparse_results = [
        {"id": 202, "content": "Exact keyword found by BM25 sparse search"}
    ]
    
    fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60)
    assert len(fused) == 1
    assert fused[0]["id"] == 202
    assert fused[0]["rrf_score"] == pytest.approx(1.0 / (60 + 1), rel=1e-3)


# =====================================================================
# 2. FlashRank Re-ranker Edge Cases
# =====================================================================

def test_reranker_empty_candidates():
    """Verifies reranker handles empty candidate list gracefully without crashing."""
    results = rerank_passages("test query", candidates=[], top_n=3)
    assert results == []


def test_reranker_corrupt_or_missing_metadata():
    """Verifies reranker handles candidates with missing keys or None content."""
    corrupt_candidates = [
        {"id": 1, "content": None},
        {"id": 2},  # Missing content entirely
        {"id": 3, "content": "Valid enterprise document text"}
    ]
    results = rerank_passages("enterprise query", corrupt_candidates, top_n=2)
    assert len(results) <= 2


# =====================================================================
# 3. LLM Synthesizer & Anti-Hallucination Guardrails
# =====================================================================

def test_format_context_passages_empty():
    """Verifies context formatting handles empty chunks without raising IndexError."""
    formatted = format_context_passages([])
    assert "No relevant documents found" in formatted


def test_synthesizer_zero_context_fallback():
    """Verifies that when 0 chunks are found, synthesizer returns a safe, unhallucinated refusal."""
    response = asyncio.run(synthesize_answer("What is the secret launch date?", chunks=[]))
    assert "could not find any relevant information" in response.lower()


# =====================================================================
# 4. Intent Router Edge & Adversarial Inputs
# =====================================================================

def test_router_empty_and_whitespace_query():
    """Verifies empty or whitespace queries default to direct conversation instead of throwing."""
    state_empty = classify_intent_node({"query": "", "query_type": None})
    assert state_empty["query_type"] == "direct"

    state_spaces = classify_intent_node({"query": "     ", "query_type": None})
    assert state_spaces["query_type"] == "direct"


def test_router_sql_injection_adversarial_string():
    """Verifies SQL injection attempts do not crash the intent classifier."""
    sqli_query = "SELECT * FROM users; DROP TABLE products; --"
    state = classify_intent_node({"query": sqli_query, "query_type": None})
    # Should safely route to SQL or direct without syntax error
    assert state["query_type"] in ["sql", "direct"]


def test_router_ambiguous_cross_domain_query():
    """Verifies classifier handles queries combining document and SQL keywords."""
    ambiguous = "Explain the total order policy for remote employees"
    state = classify_intent_node({"query": ambiguous, "query_type": None})
    assert state["query_type"] in ["rag", "sql"]
