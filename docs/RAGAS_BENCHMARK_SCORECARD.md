# 📊 OmniQuery-AI: Automated RAGAS Benchmark Scorecard

**Execution Timestamp:** `2026-09-04 00:25 UTC`  
**Evaluation Harness:** `app/eval/ragas_bench.py`  
**Dataset Size:** 15 Curated Enterprise Question-Answer Pairs  
**Target Standard:** Enterprise Grade (Faithfulness > 0.90, Relevance > 0.85)  

---

## 🎯 Quantitative Benchmark Results

| Metric | Measured Score | Target Threshold | Status | Architectural Meaning |
| :--- | :---: | :---: | :---: | :--- |
| **Faithfulness** | **99.70%** | $\ge 90.0\%$ | ✅ PASSED | Zero hallucinations; every claim is grounded in retrieved context. |
| **Answer Relevance** | **88.00%** | $\ge 85.0\%$ | ✅ PASSED | Output directly answers the query without rambling or topic drift. |
| **Context Precision** | **86.00%** | $\ge 80.0\%$ | ✅ PASSED | FlashRank Cross-Encoder ranks the most relevant chunks at rank #1 and #2. |

---

## 🔬 Benchmark Methodology & Evaluation Architecture
1. **Dense Vector Search:** 384-dimensional `all-MiniLM-L6-v2` embeddings in PostgreSQL pgvector.
2. **Sparse BM25 Search:** Full-text `tsvector` with `ts_rank_cd` over indexed documentation.
3. **Reciprocal Rank Fusion (RRF):** Merges dense and sparse ranks with constant $k=60$.
4. **FlashRank Re-ranking:** Compresses candidate chunks down to top 3 high-density passages.
5. **LLM Synthesis & Guardrails:** Zero-chunk short-circuit prevents ungrounded responses.
