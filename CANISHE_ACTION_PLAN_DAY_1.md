# 🚀 Canishe's Daily Action Plan & Concept Learning Guide: Day 1 (Week 1)

---

## 🎯 Today's Primary Objectives

1. **Pull the latest codebase** and understand the **Week 1 Core Retrieval Engine** we pushed today.
2. **Master the 4 core GenAI concepts** (RRF, SKUs & Alphanumeric Blindspots, Intent Routing, Dual Indexing).
3. **Execute local database ingestion** (`document_chunks` with `vector` + `tsvector`) and test the **Streamlit Web UI**.
4. **Build a Database Seeding Script (`app/db_seed.py`)** to populate sample `products`, `customers`, and `orders` for our upcoming Text-to-SQL engine.

---

## 📚 Phase 1: Theory & Concept Study (30–45 Mins)

Open these 4 markdown guides in your VS Code editor. Read each one carefully and understand the real-world analogies:

1. 📖 [**`docs/09_CONCEPT_SKUS_AND_ALPHANUMERIC_BLINDSPOTS.md`**](file:///Users/jnarayanassamy/personal/ai/canishe/OmniQuery-AI/docs/09_CONCEPT_SKUS_AND_ALPHANUMERIC_BLINDSPOTS.md)
   * **What to learn:** Why dense vector models (like OpenAI or HuggingFace) confuse `SKU-4001` with `SKU-4002` (0.98 similarity), and why combining BM25 keyword search + relational SQL solves catalog hallucinations.
   
2. 📖 [**`docs/10_CONCEPT_INTENT_CLASSIFICATION_AND_ROUTING.md`**](file:///Users/jnarayanassamy/personal/ai/canishe/OmniQuery-AI/docs/10_CONCEPT_INTENT_CLASSIFICATION_AND_ROUTING.md)
   * **What to learn:** How the LangGraph Classifier Node decides between `rag`, `sql`, and `direct` using Keyword Heuristics, Semantic Vector Routers, and LLM Structured JSON.

3. 📖 [**`docs/11_CONCEPT_RECIPROCAL_RANK_FUSION_WORKED_EXAMPLES.md`**](file:///Users/jnarayanassamy/personal/ai/canishe/OmniQuery-AI/docs/11_CONCEPT_RECIPROCAL_RANK_FUSION_WORKED_EXAMPLES.md)
   * **What to learn:** Why you cannot simply add dense and sparse raw scores together (apples vs. oranges), and how the formula $\text{RRF Score} = \sum \frac{1}{60 + \text{rank}}$ rewards cross-engine agreement.

4. 📖 [**`docs/12_CONCEPT_PDF_INGESTION_AND_DUAL_INDEXING.md`**](file:///Users/jnarayanassamy/personal/ai/canishe/OmniQuery-AI/docs/12_CONCEPT_PDF_INGESTION_AND_DUAL_INDEXING.md)
   * **What to learn:** How PDF text is parsed page-by-page, chunked into 500-token blocks with 50-token overlap, and stored in PostgreSQL with both `vector(384)` and `tsvector` columns.

---

## 💻 Phase 2: Hands-On Technical Tasks

### Task 1: Pull Latest Changes & Run Automated Tests (5 Mins)

In your Windows 11 terminal (PowerShell / Command Prompt):

```bash
# 1. Navigate to your project directory
cd OmniQuery-AI

# 2. Pull the latest code from GitHub
git pull origin main

# 3. Activate your virtual environment
.\.venv\Scripts\activate

# 4. Run the automated Pytest suite
pytest tests/test_hybrid_rag.py -v
```

👉 **Expected Output:** All 4 tests should pass:
- `test_chunk_documents PASSED`
- `test_reciprocal_rank_fusion PASSED`
- `test_reranker_passages PASSED`
- `test_langgraph_intent_router PASSED`

---

### Task 2: Start PostgreSQL & Run Document Ingestion (10 Mins)

1. Ensure Docker Desktop is running on Windows 11.
2. Start the PostgreSQL 16 container with `pgvector`:
   ```bash
   docker-compose up -d
   ```
3. Run the enterprise knowledge base ingestion pipeline:
   ```bash
   python -m app.rag.ingest
   ```
   * *What this does:* Loads 3 enterprise policy documents (IT Security, Remote Work, SLA Terms), chunks them, generates 384-dimensional dense vectors, and inserts them into PostgreSQL with automatic `to_tsvector` indexing.

---

### Task 3: Build the Relational Database Seeder (`app/db_seed.py`) (30 Mins)

To prepare for our **Week 2 Text-to-SQL Engine**, create a new script named `app/db_seed.py` that populates sample data into the [`customers`](file:///Users/jnarayanassamy/personal/ai/canishe/OmniQuery-AI/app/models.py#L62), [`products`](file:///Users/jnarayanassamy/personal/ai/canishe/OmniQuery-AI/app/models.py#L78), and [`orders`](file:///Users/jnarayanassamy/personal/ai/canishe/OmniQuery-AI/app/models.py#L95) tables.

Create `app/db_seed.py` with the following starter code and run it:

```python
"""
Database Seeder Script
Populates sample Enterprise Customers, Products (with unique SKUs), and Orders
for the Text-to-SQL Copilot Engine.
"""

import asyncio
from sqlalchemy import text
from app.database import AsyncSessionLocal, init_db
from app.models import Customer, Product, Order, OrderItem

async def seed_database():
    await init_db()
    
    async with AsyncSessionLocal() as session:
        # Check if already seeded
        result = await session.execute(text("SELECT count(*) FROM products;"))
        count = result.scalar()
        if count and count > 0:
            print(f"Database already contains {count} products. Skipping seeding.")
            return

        print("Seeding sample enterprise products, customers, and orders...")
        
        # 1. Add Sample Products with SKUs
        p1 = Product(sku="SKU-SRV-101", name="OmniCloud Enterprise Server 1U", category="Hardware", price=2499.00, stock_quantity=45)
        p2 = Product(sku="SKU-SW-202", name="OmniQuery AI Analytics Pro License", category="Software", price=499.00, stock_quantity=500)
        p3 = Product(sku="SKU-SEC-303", name="Hardware Security Key (FIDO2/MFA)", category="Security", price=55.00, stock_quantity=250)
        p4 = Product(sku="SKU-ACC-404", name="Ergonomic Executive Office Chair", category="Furniture", price=350.00, stock_quantity=80)
        session.add_all([p1, p2, p3, p4])
        await session.flush()

        # 2. Add Sample Customers
        c1 = Customer(name="Acme Financial Corp", email="procurement@acmefin.com", country="USA", tier="Enterprise")
        c2 = Customer(name="Bangalore AI Labs", email="admin@bangaloreai.in", country="India", tier="Gold")
        c3 = Customer(name="Dallas Logistics LLC", email="ops@dallaslogistics.com", country="USA", tier="Standard")
        session.add_all([c1, c2, c3])
        await session.flush()

        # 3. Add Sample Orders
        o1 = Order(customer_id=c1.id, status="Completed", total_amount=5497.00)
        o2 = Order(customer_id=c2.id, status="Completed", total_amount=1497.00)
        o3 = Order(customer_id=c3.id, status="Pending", total_amount=350.00)
        session.add_all([o1, o2, o3])
        await session.flush()

        # 4. Add Order Items
        item1 = OrderItem(order_id=o1.id, product_id=p1.id, quantity=2, unit_price=2499.00)
        item2 = OrderItem(order_id=o1.id, product_id=p2.id, quantity=1, unit_price=499.00)
        item3 = OrderItem(order_id=o2.id, product_id=p2.id, quantity=3, unit_price=499.00)
        item4 = OrderItem(order_id=o3.id, product_id=p4.id, quantity=1, unit_price=350.00)
        session.add_all([item1, item2, item3, item4])

        await session.commit()
        print("✅ Sample database records successfully seeded!")

if __name__ == "__main__":
    asyncio.run(seed_database())
```

Run the seeder:
```bash
python -m app.db_seed
```

---

### Task 4: Run the Backend & Streamlit Web UI (15 Mins)

1. In Terminal 1, start the FastAPI Backend:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
   * Open Swagger docs: `http://localhost:8000/docs`

2. In Terminal 2, start the Streamlit Copilot UI:
   ```bash
   streamlit run ui/streamlit_app.py --server.port 8501
   ```
   * Open UI: `http://localhost:8501`

3. Try asking these test queries in the UI and observe the routing behavior:
   * 📄 *"What is the monthly internet reimbursement allowance for remote employees?"* (Should route to `RAG` and cite `OmniCorp_Remote_Work_Guidelines.pdf`).
   * 📄 *"What does error code ERR_AUTH_401 mean?"* (Should route to `RAG` and cite `OmniCorp_IT_Security_Policy.pdf`).
   * 📊 *"How many total orders are in Completed status?"* (Should route to `SQL`).
   * 💬 *"Hi, who are you and what can you do?"* (Should route to `DIRECT`).

---

## 🎤 Phase 3: Self-Check Interview Quiz (Test Your Understanding)

Before our next pair-programming call, practice answering these 3 questions out loud:

1. **Question 1:** *"Why can't we just add the dense cosine similarity score and the BM25 score together?"*
   * *Target Answer:* Because cosine scores are bounded between 0 and 1, while BM25 scores are unbounded numbers like 15 or 20. Adding them directly causes BM25 to overpower vector search. RRF solves this by ranking positions ($1 / (k + \text{rank})$) instead of raw scores.

2. **Question 2:** *"What is the alphanumeric blindspot in AI embeddings?"*
   * *Target Answer:* Embedding models compress text into semantic ideas, so different SKUs (like `SKU-101` and `SKU-102`) or error codes look nearly identical in vector space (~0.98 similarity). We solve this using BM25 keyword indices and relational SQL tables.

3. **Question 3:** *"How does LangGraph route between RAG and SQL?"*
   * *Target Answer:* It uses an `AgentState` dictionary and a Classifier Node that evaluates whether the user's intent is document search, aggregate metrics calculation, or general conversation, then dynamically triggers the appropriate node.

---

### 📤 Submission:
When you finish your tasks, commit and push your `app/db_seed.py` to GitHub:
```bash
git add .
git commit -m "feat(db): add database seed script with sample products, customers, and orders"
git push origin main
```

Good luck Canishe! Let's connect on Google Meet for our next session.
