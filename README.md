# OmniQuery-AI: Enterprise Hybrid RAG & Autonomous SQL Copilot

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL pgvector](https://img.shields.io/badge/PostgreSQL-pgvector-blue.svg)](https://github.com/pgvector/pgvector)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![RAGAS Evaluated](https://img.shields.io/badge/Evaluated%20with-RAGAS-purple.svg)](https://github.com/explodinggradients/ragas)

A production-grade applied GenAI microservice and interactive copilot demonstrating **Hybrid Search (Dense pgvector + Sparse BM25)**, **Cross-Encoder Re-ranking**, **LangGraph Dynamic Agent Routing**, **PostgreSQL Text-to-SQL**, and **Automated RAGAS Quality Evaluation**.

---

## 🌟 Key Features

1. **Hybrid Retrieval Engine:**
   - Vector similarity search via `pgvector` (Cosine distance).
   - Full-text keyword search via PostgreSQL `tsvector` (`BM25`).
   - Merged with **Reciprocal Rank Fusion (RRF)**.
   - Context compression via **FlashRank / Cross-Encoder Re-ranking**.
2. **Text-to-SQL Autonomous Copilot:**
   - Natural language to parameterized SQL generation.
   - Schema validation and read-only execution security sandbox.
3. **LangGraph Agentic State Machine:**
   - Multi-agent router deciding between Document RAG, SQL Database queries, and Direct LLM synthesis.
4. **FastAPI Async Token Streaming:**
   - Server-Sent Events (`SSE`) for instantaneous token-by-token streaming.
5. **RAGAS Benchmark Suite:**
   - Automated quality scorecard measuring Faithfulness, Answer Relevance, and Context Precision.

---

## 🚀 Quick Start (Local Setup)

### 1. Start PostgreSQL with pgvector (via Docker)
```bash
docker-compose up -d
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables
```bash
cp .env.example .env
# Add your Gemini API Key / OpenAI Key / Ollama URL in .env
```

### 4. Run the FastAPI Backend
```bash
uvicorn app.main:app --reload --port 8000
```
* Interactive API Documentation (Swagger): [http://localhost:8000/docs](http://localhost:8000/docs)

### 5. Run the Streamlit UI
```bash
streamlit run ui/streamlit_app.py --server.port 8501
```
* Interactive Chat UI: [http://localhost:8501](http://localhost:8501)

---

## 🏗️ Repository Architecture

```
OmniQuery-AI/
├── app/
│   ├── main.py                # FastAPI Application & Streaming Endpoints
│   ├── database.py            # PostgreSQL + pgvector async connection
│   ├── rag/
│   │   ├── hybrid_retriever.py# Dense + Sparse BM25 + Cross-Encoder Reranker
│   │   └── ingest.py          # Document loader, chunker & embedder
│   ├── agents/
│   │   ├── router.py          # LangGraph state machine router
│   │   └── sql_agent.py       # Safe Text-to-SQL generation engine
│   └── eval/
│       └── ragas_bench.py     # Automated RAGAS evaluation runner
├── ui/
│   └── streamlit_app.py       # Interactive Chatbot & SQL visualizer
├── docker-compose.yml         # Containerized PostgreSQL 16 + pgvector
├── requirements.txt           # Python dependencies
└── README.md
```
