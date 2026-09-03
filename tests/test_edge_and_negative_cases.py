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


@pytest.mark.asyncio
async def test_synthesizer_zero_context_fallback():
    """Verifies that when 0 chunks are found, synthesizer returns a safe, unhallucinated refusal."""
    response = await synthesize_answer("What is the secret launch date?", chunks=[])
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


def test_router_avoids_substring_collisions():
    """Verifies router doesn't trigger on substring collisions like 'fromage' or 'borders'."""
    state_fromage = classify_intent_node({"query": "Tell me about French fromage", "query_type": None})
    assert state_fromage["query_type"] == "direct"

    state_borders = classify_intent_node({"query": "What are the borders of Texas?", "query_type": None})
    assert state_borders["query_type"] == "direct"


# =====================================================================
# 5. FastAPI Ingress & Pydantic Validation Tests
# =====================================================================

def test_api_empty_query_returns_422():
    """Verifies that empty or whitespace query returns 422 Unprocessable Entity."""
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    res_empty = client.post("/api/v1/query", json={"query": ""})
    assert res_empty.status_code == 422

    res_spaces = client.post("/api/v1/query", json={"query": "    "})
    assert res_spaces.status_code == 422
    assert "cannot be empty" in res_spaces.text


def test_api_oversized_query_returns_422():
    """Verifies that query exceeding 1000 characters is rejected with 422."""
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    oversized = "a" * 1001
    res = client.post("/api/v1/query", json={"query": oversized})
    assert res.status_code == 422


def test_api_valid_query_structure():
    """Verifies that a valid conversational query returns 200 with proper QueryResponse structure."""
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    res = client.post("/api/v1/query", json={"query": "Hello"})
    assert res.status_code == 200
    data = res.json()
    assert data["query"] == "Hello"
    assert "route_selected" in data
    assert "response" in data

