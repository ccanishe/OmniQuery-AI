"""
Document Ingestion and Semantic Chunking Pipeline
Processes multi-page PDFs and text documents:
1. Loads and parses documents page-by-page.
2. Performs semantic chunking via RecursiveCharacterTextSplitter (chunk_size=500, overlap=50).
3. Generates 384-dimensional dense vector embeddings using sentence-transformers/all-MiniLM-L6-v2.
4. Populates PostgreSQL document_chunks table with both pgvector embeddings and tsvector keyword data.
"""

import os
import sys
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from pypdf import PdfReader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from app.database import AsyncSessionLocal, init_db

# Default local embedding model (384 dimensions, fast inference on CPU/GPU)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model_instance: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy-loads and caches the sentence-transformers model instance."""
    global _model_instance
    if _model_instance is None:
        print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
        _model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model_instance


def load_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from a PDF document with metadata.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")

    reader = PdfReader(file_path)
    doc_name = os.path.basename(file_path)
    pages_data = []

    for idx, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages_data.append({
                "content": page_text.strip(),
                "metadata": {
                    "document_name": doc_name,
                    "document_id": doc_name.replace(" ", "_").lower(),
                    "page_number": idx + 1,
                    "total_pages": len(reader.pages),
                    "source": file_path
                }
            })
    return pages_data


def chunk_documents(
    documents: List[Dict[str, Any]], 
    chunk_size: int = 500, 
    chunk_overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Splits document pages into fixed-size semantic chunks with overlap.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []
    chunk_counter = 0

    for doc in documents:
        raw_text = doc["content"]
        base_meta = doc.get("metadata", {})
        split_texts = text_splitter.split_text(raw_text)

        for sub_idx, chunk_text in enumerate(split_texts):
            chunk_counter += 1
            all_chunks.append({
                "document_id": base_meta.get("document_id", "unknown_doc"),
                "document_name": base_meta.get("document_name", "Unknown Document"),
                "chunk_index": chunk_counter,
                "content": chunk_text,
                "metadata": {
                    **base_meta,
                    "chunk_sub_index": sub_idx,
                    "char_count": len(chunk_text),
                    "created_at": datetime.utcnow().isoformat()
                }
            })
    return all_chunks


def generate_embeddings(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generates 384-dimensional dense vector embeddings for each text chunk.
    """
    model = get_embedding_model()
    texts = [chunk["content"] for chunk in chunks]
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


async def insert_chunks_to_db(chunks: List[Dict[str, Any]]) -> int:
    """
    Persists document chunks into PostgreSQL with pgvector embeddings and tsvector full-text index.
    """
    if not chunks:
        return 0

    async with AsyncSessionLocal() as session:
        insert_query = text("""
            INSERT INTO document_chunks (
                document_id, document_name, chunk_index, content, embedding, tsv_content, metadata_json
            ) VALUES (
                :document_id, :document_name, :chunk_index, :content, 
                :embedding::vector, 
                to_tsvector('english', :content), 
                CAST(:metadata_json AS jsonb)
            );
        """)

        import json
        for chunk in chunks:
            await session.execute(
                insert_query,
                {
                    "document_id": chunk["document_id"],
                    "document_name": chunk["document_name"],
                    "chunk_index": chunk["chunk_index"],
                    "content": chunk["content"],
                    "embedding": str(chunk["embedding"]),
                    "metadata_json": json.dumps(chunk["metadata"])
                }
            )
        await session.commit()

    print(f"Successfully ingested {len(chunks)} chunks into PostgreSQL.")
    return len(chunks)


async def ingest_pipeline(file_path: Optional[str] = None, raw_docs: Optional[List[Dict[str, Any]]] = None) -> int:
    """
    Full ingestion pipeline orchestrator: Load -> Chunk -> Embed -> Database Insert.
    """
    await init_db()

    docs = []
    if file_path:
        if file_path.endswith(".pdf"):
            docs = load_pdf(file_path)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            doc_name = os.path.basename(file_path)
            docs = [{
                "content": content,
                "metadata": {
                    "document_name": doc_name,
                    "document_id": doc_name.replace(" ", "_").lower(),
                    "source": file_path
                }
            }]
    elif raw_docs:
        docs = raw_docs
    else:
        docs = get_sample_enterprise_docs()

    chunks = chunk_documents(docs)
    embedded_chunks = generate_embeddings(chunks)
    total_saved = await insert_chunks_to_db(embedded_chunks)
    return total_saved


def get_sample_enterprise_docs() -> List[Dict[str, Any]]:
    """
    Generates rich enterprise seed documentation for instant zero-dependency testing.
    """
    return [
        {
            "content": """
OmniCorp Global IT Security & Data Access Policy (Policy Ref: SEC-2026-V4)
1. Multi-Factor Authentication (MFA): All employees and contractors must use hardware security keys or authenticator apps (TOTP) for all corporate logins. SMS-based authentication is strictly prohibited due to SIM-swapping vulnerabilities.
2. Data Retention & Privacy: Customer PII (Personally Identifiable Information) must be encrypted at rest using AES-256 and in transit using TLS 1.3. Logs containing customer query data must be purged after 90 days unless subject to legal hold (Clause 4.2).
3. Incident Response Protocol: In the event of a suspected security breach, employees must report to soc@omnicorp.internal within 15 minutes. The Severity-1 triage team will initiate incident containment within 30 minutes.
4. Error Codes & Diagnostic Logs: System error ERR_AUTH_401 indicates invalid bearer token credentials. Error ERR_CONN_503 indicates PostgreSQL connection pool exhaustion.
            """.strip(),
            "metadata": {
                "document_name": "OmniCorp_IT_Security_Policy.pdf",
                "document_id": "sec_policy_2026",
                "department": "Security",
                "classification": "Internal Only"
            }
        },
        {
            "content": """
OmniCorp Remote Work & Reimbursement Guidelines (HR-GUIDE-2026)
1. Eligibility: Full-time employees who have completed their 90-day onboarding period are eligible for flexible hybrid and remote work arrangements.
2. Equipment Allowance: A one-time home office stipend of $1,500 is provided to full-time remote engineers to purchase ergonomic chairs, monitors, and noise-cancelling headsets.
3. Internet Reimbursement: Employees can expense up to $75 per month for high-speed home broadband through the Expensify portal before the 25th of each calendar month.
4. Core Collaboration Hours: Distributed engineering teams across Central Time (Dallas) and Indian Standard Time (Bangalore) must align on daily overlap hours between 8:00 AM – 10:30 AM Central Time (6:30 PM – 9:00 PM IST) for pair programming and standups.
            """.strip(),
            "metadata": {
                "document_name": "OmniCorp_Remote_Work_Guidelines.pdf",
                "document_id": "hr_remote_2026",
                "department": "Human Resources",
                "classification": "Internal Only"
            }
        },
        {
            "content": """
OmniCorp Customer Returns, Warranty & SLA Terms (SLA-DOC-802)
1. Standard Return Window: Enterprise hardware products can be returned within 30 days of delivery for a 100% refund, provided items are returned in original packaging.
2. Cloud Service Level Agreement (SLA): OmniQuery-AI cloud services guarantee 99.95% monthly uptime. If uptime drops below 99.95%, enterprise customers receive a 10% credit. If uptime drops below 99.0%, a 25% credit is issued.
3. Support Tiers: Platinum Enterprise tier customers receive 24/7 dedicated engineering support with guaranteed response times under 15 minutes for P0/P1 outages. Gold tier customers receive 1-hour response times during standard business hours.
            """.strip(),
            "metadata": {
                "document_name": "OmniCorp_Customer_SLA_Terms.pdf",
                "document_id": "sla_terms_802",
                "department": "Legal & Operations",
                "classification": "Public"
            }
        }
    ]


if __name__ == "__main__":
    file_arg = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"Starting OmniQuery-AI Ingestion Pipeline...")
    asyncio.run(ingest_pipeline(file_arg))
