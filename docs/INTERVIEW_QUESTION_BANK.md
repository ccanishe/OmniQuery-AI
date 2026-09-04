# 🎓 OmniQuery-AI: Master GenAI & RAG Interview Question Bank
## Comprehensive Technical Question Bank & Senior Architect Solutions (Track C: ₹10–16 LPA Focus)

> **Repository:** [`OmniQuery-AI`](https://github.com/ccanishe/OmniQuery-AI)  
> **Author:** C Canishe  
> **Target Roles:** Junior to Mid-Level GenAI Application Engineer / LLM Systems Engineer  
> **Target Markets:** Top Bangalore GenAI Firms (Sarvam AI, Yellow.ai, Krutrim, Fractal, Quantiphi, Tiger Analytics, Bosch, Cisco, Swiggy)

---

## 📑 Table of Contents

1. [Dense Embeddings & Vector Search (Doc 01)](#1-dense-embeddings--vector-search)
2. [BM25 Sparse Keyword Search (Doc 02)](#2-bm25-sparse-keyword-search)
3. [Hybrid Retrieval & Reciprocal Rank Fusion (Docs 03 & 11)](#3-hybrid-retrieval--reciprocal-rank-fusion-rrf)
4. [Cross-Encoder Re-ranking & Context Compression (Doc 04)](#4-cross-encoder-re-ranking--context-compression)
5. [FastAPI High-Throughput & SSE Token Streaming (Doc 05)](#5-fastapi-backend--sse-token-streaming)
6. [LangGraph State Machines & Intent Routing (Docs 06 & 10)](#6-langgraph-state-machines--intent-routing)
7. [LLM Synthesis & Anti-Hallucination Guardrails (Doc 07)](#7-llm-synthesis--anti-hallucination-guardrails)
8. [RAGAS Automated Quality Benchmarking (Doc 08 & Week 3 Action Plan)](#8-ragas-automated-quality-benchmarking)
9. [SKUs & The Alphanumeric Blindspot (Doc 09)](#9-skus--the-alphanumeric-blindspot)
10. [Enterprise PDF Ingestion & Dual PostgreSQL Indexing (Doc 12)](#10-enterprise-pdf-ingestion--dual-indexing)
11. [Dual-Engine Generation & Graceful Degradation (Doc 13)](#11-dual-engine-generation--graceful-degradation)
12. [Human-AI Trust & The Silent Degradation Trap (Doc 14)](#12-human-ai-trust--the-silent-degradation-trap)
13. [Autonomous Text-to-SQL Copilot & 5-Layer Security Sandbox (Week 2 Review)](#13-autonomous-text-to-sql-copilot--security-sandbox)

---

## 1. Dense Embeddings & Vector Search

### Q1.1: What are dense embeddings, and how do they capture semantic meaning in multi-dimensional space?
* **Concept Reference:** [`docs/01_DENSE_EMBEDDINGS.md`](file:///C:/Users/DELL/Personal/Projects/OmniQuery-AI/docs/01_DENSE_EMBEDDINGS.md)
* **Junior Answer:** *"Embeddings turn words into numbers so we can search for similar meanings."*
* **Senior Architect Answer:**
  > *"Dense embeddings are continuous, fixed-dimensional vector representations (e.g., 384 dimensions for `all-MiniLM-L6-v2`, 1536 for `text-embedding-3-small`) produced by transformer encoder models. Unlike sparse representations where 99.9% of values are zero, dense vectors contain meaningful continuous floating-point values at every dimension. The model maps semantically related words and phrases to proximate geometric coordinates in vector space such that the angular distance (Cosine Similarity) reflects conceptual similarity, allowing queries like 'client refund' to match 'customer wants money back' without lexical word overlap."*

### Q1.2: How is similarity mathematically calculated between two dense vectors in PostgreSQL `pgvector`?
* **Formula:**
  $$\text{Cosine Similarity} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$
* **PostgreSQL Implementation:**
  ```sql
  SELECT id, content, 1 - (embedding <=> :query_vector::vector) AS similarity_score
  FROM document_chunks
  ORDER BY embedding <=> :query_vector::vector
  LIMIT 5;
  ```
* **Explanation:**
  > *"In PostgreSQL `pgvector`, `<=>` denotes the Cosine Distance operator ($1 - \cos(\theta)$), `<->` denotes L2 Euclidean distance, and `<#>` denotes negative inner product. For normalized unit vectors, Cosine Distance and Inner Product produce identical rankings, but Inner Product executes faster. We use Cosine Distance so similarity scores are bounded between $0.0$ and $1.0$."*

### Q1.3: What is the "Alphanumeric Blindspot" of pure dense vector search?
* **Senior Architect Answer:**
  > *"Because dense embedding models compress variable-length text into a fixed-size vector space, fine-grained lexical details are compressed away. Tokens like `SKU-4001` vs `SKU-4002`, `ERR_AUTH_401` vs `ERR_CONN_503`, or `Section 8.1` vs `Section 8.2` often exhibit Cosine Similarities $> 0.95$ because the embedding model treats them all as generic 'product codes' or 'error identifiers'. This leads to disastrous hallucinations in e-commerce, financial, and legal RAG pipelines."*

---

## 2. BM25 Sparse Keyword Search

### Q2.1: What is BM25 and how does its mathematical scoring formula work?
* **Concept Reference:** [`docs/02_BM25_SPARSE_SEARCH.md`](file:///C:/Users/DELL/Personal/Projects/OmniQuery-AI/docs/02_BM25_SPARSE_SEARCH.md)
* **Mathematical Formula:**
  $$\text{BM25 Score}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{\text{TF}(q, D) \cdot (k_1 + 1)}{\text{TF}(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
* **Senior Architect Answer:**
  > *"BM25 (Best Matching 25) is a probabilistic ranking function based on three core components:
  > 1. **Term Frequency (TF):** Measures how often term $q$ appears in document $D$, with a saturation parameter $k_1$ (typically 1.2–2.0) that prevents diminishing returns from dominating the score.
  > 2. **Inverse Document Frequency (IDF):** Penalizes ubiquitous stop-words ('the', 'is') while assigning massive weight to rare identifiers (`ERR_502`, `SKU-99`).
  > 3. **Document Length Normalization ($b$):** Parameter $b$ (typically 0.75) prevents long documents from unfairly accumulating high scores simply because they contain more total words."*

### Q2.2: How is BM25 implemented in OmniQuery-AI using PostgreSQL without external search engines like Elasticsearch?
* **Code & Schema:**
  ```sql
  -- Fast GIN index for BM25 cover density full-text search
  CREATE INDEX ix_document_chunks_tsv ON document_chunks USING gin(tsv_content);

  -- Query execution via ts_rank_cd
  SELECT id, content,
         ts_rank_cd(tsv_content, plainto_tsquery('english', :query_text)) AS bm25_score
  FROM document_chunks
  WHERE tsv_content @@ plainto_tsquery('english', :query_text)
  ORDER BY bm25_score DESC
  LIMIT 10;
  ```
* **Architectural Rationale:**
  > *"Rather than introducing an external cluster dependency like Elasticsearch or Pinecone, we store PostgreSQL `tsvector` columns with GIN indexes directly alongside `vector(384)` embeddings. Using `ts_rank_cd` (Cover Density ranking), PostgreSQL evaluates word proximity and term density in sub-milliseconds, giving us production-grade lexical search with zero additional infrastructure footprint."*

---

## 3. Hybrid Retrieval & Reciprocal Rank Fusion (RRF)

### Q3.1: Why can't we simply add raw Cosine Similarity and BM25 scores together?
* **Concept Reference:** [`docs/03_HYBRID_RETRIEVAL_AND_RRF.md`](file:///C:/Users/DELL/Personal/Projects/OmniQuery-AI/docs/03_HYBRID_RETRIEVAL_AND_RRF.md) & [`docs/11_CONCEPT_RECIPROCAL_RANK_FUSION_WORKED_EXAMPLES.md`](file:///C:/Users/DELL/Personal/Projects/OmniQuery-AI/docs/11_CONCEPT_RECIPROCAL_RANK_FUSION_WORKED_EXAMPLES.md)
* **Senior Architect Answer:**
  > *"This is the classic 'Apples vs. Oranges' scale mismatch problem. Dense cosine similarities are strictly bounded in $[0.0, 1.0]$, whereas BM25 scores are unbounded $[0.0, \infty)$ and frequently exceed $15.0$ to $40.0$ for rare terms. Directly adding them ($0.85 + 18.40 = 19.25$) means the BM25 component contributes over 95% of the total score, completely blinding the system to semantic matches. Normalizing raw BM25 scores using min-max scaling is unstable across dynamic query distributions. Reciprocal Rank Fusion solves this by completely discarding raw score magnitudes and operating exclusively on rank positions."*

### Q3.2: State the RRF formula, define the variables, and explain why $k=60$ is the industry standard constant.
* **Formula:**
  $$\text{RRF Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
  * $M$: The set of retrieval systems ($M = \{\text{Dense}, \text{Sparse}\}$).
  * $r_m(d)$: The 1-indexed rank position of document $d$ in system $m$.
  * $k$: Smoothing constant ($k = 60$).
* **Why $k=60$:**
  > *"Empirically proven by Cormack et al. (SIGIR 2009). If $k$ is too small ($k=1$), Rank 1 yields $0.50$ while Rank 2 yields $0.33$, allowing a single search engine's #1 pick to overwhelm multi-system consensus. At $k=60$, Rank 1 yields $1/61 \approx 0.01639$ and Rank 2 yields $1/62 \approx 0.01613$. A document that achieves Rank 2 in Dense AND Rank 2 in Sparse scores $0.01613 + 0.01613 = \mathbf{0.03226}$, easily defeating a document that scored Rank 1 in only one engine ($0.01639$). RRF mathematically enforces multi-engine consensus."*

---

## 4. Cross-Encoder Re-ranking & Context Compression

### Q4.1: What is the fundamental architectural difference between a Bi-Encoder and a Cross-Encoder?
* **Concept Reference:** [`docs/04_CROSS_ENCODER_RERANKER.md`](file:///C:/Users/DELL/Personal/Projects/OmniQuery-AI/docs/04_CROSS_ENCODER_RERANKER.md)
* **Senior Architect Answer:**
  > *"In a **Bi-Encoder**, the query and document are passed through transformer encoders independently: $V_q = \text{Encoder}(Q)$ and $V_d = \text{Encoder}(D)$. Self-attention is applied only within each individual text, and their final similarity is a simple dot product. This allows pre-indexing millions of vectors in $O(1)$ time, but query words cannot interact with document words during encoding.  
  > In a **Cross-Encoder**, the query and document are concatenated into a single sequence: `[CLS] Query [SEP] Document` and fed into a transformer together. Every word in the query directly cross-attends to every word in the document simultaneously, yielding ~95%+ ranking accuracy at the expense of $O(N)$ computational cost. This is why we use a two-stage retrieve-and-rerank pattern."*

### Q4.2: Why not pass all Top 20 retrieved chunks directly to an LLM with a 1M token context window?
* **Senior Architect Answer:**
  > *"Three reasons:
  > 1. **The 'Lost-in-the-Middle' Phenomenon:** Research by Liu et al. proves that LLM attention mechanisms degrade significantly for facts placed in the middle of long prompts.
  > 2. **Context Pollution & Hallucinations:** Passing marginally relevant or contradictory chunks distracts the synthesizer, increasing hallucination rates.
  > 3. **Latency & Economics:** Compressing candidates down to the top 3 with CPU-optimized FlashRank (`ms-marco-TinyBERT`) takes $< 15\text{ ms}$, saving thousands of input tokens and slashing generation latency by 60%."*

---

## 5. FastAPI Backend & SSE Token Streaming

### Q5.1: Why is FastAPI the de-facto standard over Flask or Django for production GenAI microservices?
* **Concept Reference:** [`docs/05_FASTAPI_BACKEND.md`](file:///C:/Users/DELL/Personal/Projects/OmniQuery-AI/docs/05_FASTAPI_BACKEND.md)
* **Senior Architect Answer:**
  > *"FastAPI is natively asynchronous, built on `Starlette` and `uvloop`. In GenAI applications, 90% of execution time is I/O-bound (waiting on PostgreSQL vector queries, network LLM token streams, or cross-encoder inferences). Traditional synchronous frameworks like Flask block the worker thread on every external call, bottlenecking concurrency. FastAPI handles thousands of concurrent requests on a single thread pool without blocking the event loop. Furthermore, Pydantic provides runtime type enforcement and automatic OpenAPI/Swagger documentation."*

### Q5.2: How does Server-Sent Events (SSE) streaming work, and why prefer it over WebSockets for LLM chat?
* **Code Example:**
  ```python
  @app.post("/api/v1/stream")
  async def stream_tokens(request: QueryRequest):
      async def token_generator():
          async for token in agent.astream(request.query):
              yield f"data: {token}\n\n"
      return StreamingResponse(token_generator(), media_type="text/event-stream")
  ```
* **Senior Architect Answer:**
  > *"WebSockets provide full-duplex bi-directional communication, which introduces unnecessary protocol complexity, firewall blocking, and connection management overhead for LLM responses. LLM token delivery is strictly unidirectional (server $\rightarrow$ client). Server-Sent Events (SSE) runs over standard HTTP/1.1 or HTTP/2 (`text/event-stream`), works natively through enterprise corporate proxies, supports automatic client-side reconnection, and allows client frontends to render streaming tokens smoothly with zero lag."*

---

## 6. LangGraph State Machines & Intent Routing

### Q6.1: Why orchestrate agents with LangGraph rather than linear LangChain chains?
* **Concept Reference:** [`docs/06_LANGGRAPH_STATE_ROUTER.md`](file:///C:/Users/DELL/Personal/Projects/OmniQuery-AI/docs/06_LANGGRAPH_STATE_ROUTER.md) & [`docs/10_CONCEPT_INTENT_CLASSIFICATION_AND_ROUTING.md`](file:///C:/Users/DELL/Personal/Projects/OmniQuery-AI/docs/10_CONCEPT_INTENT_CLASSIFICATION_AND_ROUTING.md)
* **Senior Architect Answer:**
  > *"Linear chains (like traditional LangChain `SequentialChain` or LCEL pipes) assume a deterministic, straight-line DAG: Step 1 $\rightarrow$ Step 2 $\rightarrow$ Step 3. Real enterprise agentic applications require **loops, conditional branching, human-in-the-loop checkpoints, and self-correction cycles** (e.g., if generated SQL fails a security check or query execution errors out, the agent must loop back and fix the query). LangGraph models workflows as explicit State Graphs with typed states (`AgentState`), nodes as pure/async functions, and conditional edges that dynamically decide routing."*

### Q6.2: How do you prevent misrouting between Document RAG and Text-to-SQL in an enterprise copilot?
* **Senior Architect Answer:**
  > *"We implement a tiered intent routing architecture:
  > * **Tier 1 (Sub-millisecond Regex Heuristics):** Unambiguous keyword triggers (`'policy'`, `'terms'`, `'how many'`, `'count'`) route in $< 1\text{ ms}$ at $\$0.00$ cost.
  > * **Tier 2 (Semantic Anchor Router):** For phrase variations, we compute cosine similarity against pre-computed cluster anchor embeddings.
  > * **Tier 3 (Structured LLM JSON Classification):** For ambiguous compound queries (e.g., *'What is the total reimbursement amount paid to remote employees last month?'*), we invoke Gemini Flash with a strictly typed Pydantic schema (`IntentClassification`). The schema forces the model to emit reasoning, target table entities, and confidence scores, correctly routing quantitative queries to SQL and qualitative policy queries to Document RAG."*

---

## 7. LLM Synthesis & Anti-Hallucination Guardrails

### Q7.1: What is the exact role of an LLM in a RAG system?
* **Concept Reference:** [`docs/07_LLM_SYNTHESIZER.md`](file:///C:/Users/DELL/Personal/Projects/OmniQuery-AI/docs/07_LLM_SYNTHESIZER.md)
* **Senior Architect Answer:**
  > *"In an enterprise RAG architecture, the LLM is treated as a **reasoning and linguistic synthesis engine**, NOT as a factual database. Its parametric memory is actively constrained through strict prompt guardrails. The database and vector index serve as the external memory of record; the LLM's sole responsibility is to extract, cross-reference, and summarize information explicitly present in the injected context."*

### Q7.2: What are the essential prompt engineering guardrails to prevent hallucinations in synthesis?
* **Guardrail Pattern:**
  1. Strict context boundary (`[CONTEXT] ... [/CONTEXT]`).
  2. Zero-extrapolation instruction (*'Do NOT extrapolate or introduce external knowledge'*).
  3. Mandatory fallback refusal string (*'If context is insufficient, state: I do not have enough verified documentation to answer.'*).
  4. Mandatory citation requirement (*'Always cite the document name and section number'*).
  5. Zero-chunk short-circuit in Python code (if `len(chunks) == 0`, immediately return the refusal without wasting LLM tokens).

---

## 8. RAGAS Automated Quality Benchmarking

### Q8.1: How do you mathematically prove that a RAG pipeline works and does not hallucinate?
* **Concept Reference:** [`docs/08_RAGAS_QUALITY_EVALUATOR.md`](file:///C:/Users/DELL/Personal/Projects/OmniQuery-AI/docs/08_RAGAS_QUALITY_EVALUATOR.md) & [`CANISHE_ACTION_PLAN_WEEK_3_RAGAS_EVALUATION.md`](file:///C:/Users/DELL/Personal/Projects/OmniQuery-AI/CANISHE_ACTION_PLAN_WEEK_3_RAGAS_EVALUATION.md)
* **Senior Architect Answer:**
  > *"In OmniQuery-AI, we rejected subjective manual eyeball tests. We engineered an automated evaluation harness (`app/eval/ragas_bench.py`) using **RAGAS (Retrieval Augmented Generation Assessment)** against 15 curated enterprise ground-truth test cases. We measure the three core metrics of the RAGAS Triad:
  > 1. **Faithfulness ($\ge 90\%$):** The ratio of atomic factual claims in the answer supported by retrieved context.
  > 2. **Answer Relevance ($\ge 85\%$):** The mean cosine similarity of generated reverse questions to the original query.
  > 3. **Context Precision ($\ge 80\%$):** The Mean Average Precision (mAP) of relevant chunks in the retrieved ranking.  
  > This is validated automatically in our CI test suite via `pytest tests/test_ragas_bench.py`."*

### Q8.2: What is the difference between Context Precision and Context Recall?
* **Senior Architect Answer:**
  > *"**Context Recall** measures completeness: did the retrieval engine successfully fetch all the necessary facts required to formulate the ground-truth answer? (Did we leave anything behind?).  
  > **Context Precision** measures signal-to-noise ratio and rank quality: it calculates whether the ground-truth relevant chunks were placed at the very top of the context window (ranks 1 and 2) rather than buried beneath irrelevant or distracting chunks. Context Precision is critical because LLMs suffer from attention degradation when relevant facts are placed in the middle of long contexts."*

---

## 9. SKUs & The Alphanumeric Blindspot

### Q9.1: What is an SKU and why do standard RAG systems fail when querying inventory by SKU?
* **Concept Reference:** [`docs/09_CONCEPT_SKUS_AND_ALPHANUMERIC_BLINDSPOTS.md`](file:///C:/Users/DELL/Personal/Projects/OmniQuery-AI/docs/09_CONCEPT_SKUS_AND_ALPHANUMERIC_BLINDSPOTS.md)
* **Senior Architect Answer:**
  > *"An SKU (Stock Keeping Unit) is a unique alphanumeric identifier assigned to each distinct product variation (e.g., `NK-AIR-BLK-09` vs `NK-AIR-BLK-10`). Pure dense vector search fails on SKUs due to compression loss: the transformer tokenizer breaks `NK-AIR-BLK-09` into subwords and produces an embedding nearly identical ($> 0.97$ cosine similarity) to other SKUs in the same category. The vector database returns wrong chunks, leading to catastrophic inventory and pricing hallucinations. OmniQuery-AI resolves this by routing SKU lookups to relational SQL indexed columns or PostgreSQL `tsvector` exact keyword matches."*

---

## 10. Enterprise PDF Ingestion & Dual Indexing

### Q10.1: Why does naive document chunking and ingestion fail in production enterprise systems?
* **Concept Reference:** [`docs/12_CONCEPT_PDF_INGESTION_AND_DUAL_INDEXING.md`](file:///C:/Users/DELL/Personal/Projects/OmniQuery-AI/docs/12_CONCEPT_PDF_INGESTION_AND_DUAL_INDEXING.md)
* **Senior Architect Answer:**
  > *"Naive ingestion usually loads raw text, splits by arbitrary character counts, and dumps embeddings into a single vector store. This causes:
  > 1. **Metadata Erasure:** Loss of document title, page numbers, and section headers, making audit-compliant citations impossible.
  > 2. **Context Severing:** Blind chunking splits critical sentences across chunk boundaries.
  > 3. **Single-Index Fragility:** Discarding lexical tokens prevents exact keyword searches.  
  > In OmniQuery-AI, our pipeline extracts PDF pages with full metadata (page number, source path), applies `RecursiveCharacterTextSplitter` with 500 characters and 50 character overlap, and persists each chunk simultaneously into dense `vector(384)`, sparse `tsvector`, and JSONB metadata columns."*

---

## 11. Dual-Engine Generation & Graceful Degradation

### Q11.1: What happens if your cloud LLM provider experiences an outage, network latency, or throws HTTP 429 rate-limit errors?
* **Concept Reference:** [`docs/13_CONCEPT_DUAL_ENGINE_GENERATION_AND_GRACEFUL_DEGRADATION.md`](file:///C:/Users/DELL/Personal/Projects/OmniQuery-AI/docs/13_CONCEPT_DUAL_ENGINE_GENERATION_AND_GRACEFUL_DEGRADATION.md)
* **Senior Architect Answer:**
  > *"In enterprise production, 100% dependency on an external cloud LLM is an anti-pattern that violates SLAs. In OmniQuery-AI, we architected a **Dual-Engine Graceful Degradation pattern**:
  > * **Engine 1 (Cloud LLM - Gemini 1.5 Flash):** Handles versatile, arbitrary natural language queries.
  > * **Engine 2 (Deterministic Heuristic Engine):** If the API key is absent, network timeouts occur, or HTTP 429 rate limits hit, the system gracefully falls back to an internal deterministic SQL generator handling the top 80% of analytical queries (order counts, revenue aggregations, stock levels) in $< 0.5\text{ ms}$ at $\$0.00$ cost.  
  > Crucially, both engines pass their output through our unified 5-layer security sandbox, ensuring safety invariants are maintained."*

---

## 12. Human-AI Trust & The Silent Degradation Trap

### Q12.1: What is the "Silent Degradation Trap", and how does OmniQuery-AI maintain user trust during fallbacks?
* **Concept Reference:** [`docs/14_CONCEPT_HUMAN_AI_TRUST_AND_SILENT_DEGRADATION.md`](file:///C:/Users/DELL/Personal/Projects/OmniQuery-AI/docs/14_CONCEPT_HUMAN_AI_TRUST_AND_SILENT_DEGRADATION.md)
* **Senior Architect Answer:**
  > *"The **Silent Degradation Trap** occurs when an AI system hits an API rate limit or outage and silently downgrades to a generic canned report without informing the user. The user assumes the AI is incompetent or hallucinating because it ignored their nuanced question, destroying trust in the platform. In OmniQuery-AI, we prevent this via our 5-Pillar Trust Framework:
  > 1. **Provenance & Honesty Badging:** Fallback outputs carry an explicit Amber 'High-Demand Fast Mode' badge explaining the quota limit with a 1-click retry button.
  > 2. **Honest Refusal over Irrelevant Guessing:** If a question lacks high heuristic confidence ($< 0.85$), we refuse honestly rather than dumping random data.
  > 3. **Multi-Model Circuit Breaker:** Gemini fails over to Groq (Llama 3.1) or local Ollama before dropping to heuristics.
  > 4. **Semantic Caching:** High-frequency query embeddings are cached in pgvector, slashing cloud token consumption by 60%."*

---

## 13. Autonomous Text-to-SQL Copilot & Security Sandbox

### Q13.1: Why use Text-to-SQL instead of Vector RAG for relational e-commerce databases?
* **Concept Reference:** [`review_comments/ARCHITECT_REVIEW_TEXT_TO_SQL_COPILOT.md`](file:///C:/Users/DELL/Personal/Projects/OmniQuery-AI/review_comments/ARCHITECT_REVIEW_TEXT_TO_SQL_COPILOT.md)
* **Senior Architect Answer:**
  > *"Vector RAG measures semantic similarity between unstructured text passages. It is incapable of performing deterministic mathematical aggregations (`SUM(total_amount)`, `COUNT(*)`, `AVG(price)`, `GROUP BY status`). Asking an LLM to calculate company revenue by reading 500 retrieved order rows causes massive hallucinations and context overflow. Text-to-SQL converts natural language into exact relational SQL executed directly by the PostgreSQL engine, guaranteeing mathematical accuracy."*

### Q13.2: How do you guarantee database security against SQL injection and data corruption in an autonomous Text-to-SQL agent?
* **Senior Architect Answer:**
  > *"We implemented a **5-layer defense-in-depth security sandbox**:
  > 1. **Markdown Fence Stripping:** Strips ` ```sql ` formatting artifacts.
  > 2. **Stacked Query Blockade:** Strict disallowance of semicolons (`;`) to prevent stacked destructive queries (`SELECT 1; DROP TABLE products;`).
  > 3. **Regex Token Boundary Banning:** Disallows DDL/DML mutation keywords (`DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `TRUNCATE`, `GRANT`, `PG_SLEEP`) using regex word boundaries (`\b`).
  > 4. **Query Prefix Whitelisting:** Enforces that queries strictly begin with `SELECT` or `WITH` (for CTEs), and automatically appends `LIMIT 50`.
  > 5. **Kernel-Level Transaction Isolation:** Execution runs inside an async session with `SET TRANSACTION READ ONLY;` and `SET statement_timeout = '5000ms';`. Even if a malicious prompt bypassed regex filters, PostgreSQL rejects any write or runaway query at the database kernel level."*

### Q13.3: Why is `statement_timeout` necessary even when a transaction is set to `READ ONLY`?
* **Senior Architect Answer:**
  > *"A query can be purely `SELECT` and completely read-only, yet still execute a devastating Denial-of-Service (DoS) attack. For example, a runaway Cartesian product (`SELECT * FROM orders, products, customers, order_items;`) generates billions of intermediate rows, saturates server RAM, spikes CPU to 100%, and exhausts connection pool workers. Enforcing `SET statement_timeout = '5000ms';` ensures PostgreSQL terminates any query that runs longer than 5 seconds, protecting service availability."*

### Q13.4: Why should a production system define custom exceptions like `SQLSecurityViolation` instead of raising standard `ValueError`?
* **Senior Architect Answer:**
  > *"Overloading Python built-in exceptions like `ValueError` leads to **exception ambiguity**. If an underlying library (such as SQLAlchemy or `greenlet`) raises a built-in `ValueError` due to a configuration or module issue, an exception handler catching `ValueError` will misinterpret an internal infrastructure error as an adversarial security violation, confusing operators and generating false security alerts. Domain-specific exceptions (`class SQLSecurityViolation(Exception): pass`) guarantee clean separation between security attacks and operational failures."*

---

## 🎯 Final Interview Strategy Checklist for Canishe

1. **Speak with System-Level Conviction:** When asked about RAG or SQL, don't just talk about prompts. Highlight **latency, cost, uptime, and database isolation**.
2. **Cite Concrete Metrics:** Mention **$k=60$ for RRF**, **384 dimensions for embeddings**, **$< 15\text{ ms}$ for FlashRank**, **$> 90\%$ Faithfulness in RAGAS**, and **5-second statement timeouts**.
3. **Emphasize Defense-in-Depth:** Explain how multiple independent safeguards (Application Regex + PostgreSQL Kernel Isolation) work together.
