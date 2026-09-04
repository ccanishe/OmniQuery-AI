"""
Automated Pytest Suite for RAGAS Evaluation Harness
Verifies that RAG quality metrics meet enterprise production thresholds.
"""

import os
import json
import pytest
from app.eval.ragas_bench import (
    calculate_ragas_scores, 
    generate_markdown_scorecard, 
    run_pipeline_for_eval
)


def test_ground_truth_dataset_integrity():
    """Verifies ground_truth.json exists and contains well-formed enterprise Q&A pairs."""
    eval_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app", "eval")
    ground_truth_path = os.path.join(eval_dir, "ground_truth.json")
    
    assert os.path.exists(ground_truth_path), "ground_truth.json does not exist"
    
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert isinstance(data, list), "ground_truth.json root must be a list"
    assert len(data) >= 10, f"Expected at least 10 evaluation test cases, found {len(data)}"
    
    required_keys = {"id", "question", "ground_truth", "category"}
    for item in data:
        missing = required_keys - set(item.keys())
        assert not missing, f"Item {item.get('id', 'unknown')} missing required keys: {missing}"
        assert len(item["question"].strip()) > 10, f"Question too short in {item['id']}"
        assert len(item["ground_truth"].strip()) > 10, f"Ground truth too short in {item['id']}"


def test_ragas_scores_meet_thresholds():
    """Verifies that RAGAS evaluation metrics meet enterprise production standards."""
    sample_eval_records = [
        {
            "question": "What is the return window?",
            "contexts": ["Items can be returned within 30 days of delivery."],
            "answer": "The return window is 30 days from delivery.",
            "ground_truth": "Items can be returned within 30 days of delivery.",
            "category": "returns"
        },
        {
            "question": "What does error code ERR_AUTH_401 mean?",
            "contexts": ["Error ERR_AUTH_401 indicates invalid bearer token credentials."],
            "answer": "ERR_AUTH_401 indicates invalid bearer token credentials.",
            "ground_truth": "ERR_AUTH_401 indicates invalid bearer token credentials.",
            "category": "troubleshooting"
        }
    ]
    
    scores = calculate_ragas_scores(sample_eval_records)
    
    assert scores["faithfulness"] >= 0.85, f"Faithfulness dropped below threshold: {scores['faithfulness']}"
    assert scores["answer_relevance"] >= 0.80, f"Answer relevance dropped below threshold: {scores['answer_relevance']}"
    assert scores["context_precision"] >= 0.75, f"Context precision dropped below threshold: {scores['context_precision']}"


def test_markdown_scorecard_generation():
    """Verifies markdown scorecard generation produces compliant markdown with status tags."""
    test_scores = {
        "faithfulness": 0.95,
        "answer_relevance": 0.88,
        "context_precision": 0.86
    }
    
    markdown = generate_markdown_scorecard(test_scores, total_samples=15)
    
    assert "# 📊 OmniQuery-AI: Automated RAGAS Benchmark Scorecard" in markdown
    assert "Faithfulness" in markdown
    assert "95.00%" in markdown
    assert "✅ PASSED" in markdown
    assert "FlashRank Re-ranking" in markdown


@pytest.mark.asyncio
async def test_end_to_end_eval_pipeline_execution():
    """Verifies run_pipeline_for_eval executes against test inputs and returns well-formed records."""
    test_questions = [
        {
            "id": "eval_test_01",
            "question": "What is the return window for hardware products?",
            "ground_truth": "Enterprise hardware products can be returned within 30 days of delivery.",
            "category": "returns"
        }
    ]
    
    eval_records = await run_pipeline_for_eval(test_questions)
    
    assert len(eval_records) == 1
    rec = eval_records[0]
    assert rec["question"] == test_questions[0]["question"]
    assert rec["ground_truth"] == test_questions[0]["ground_truth"]
    assert "contexts" in rec and len(rec["contexts"]) > 0
    assert "answer" in rec and len(rec["answer"]) > 10
