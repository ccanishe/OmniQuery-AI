"""
SQLAlchemy Database Models for OmniQuery-AI
Defines schema for:
1. Unstructured Document Chunks (Hybrid RAG: Dense pgvector + Sparse tsvector)
2. Structured Relational Data (Enterprise Text-to-SQL: Customers, Products, Orders)
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    Column, Integer, String, Text, Float, DateTime, ForeignKey, Index, func
)
from sqlalchemy.orm import declarative_base, relationship, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import TSVECTOR, JSONB
from pgvector.sqlalchemy import Vector

from app.database import Base


# ==========================================
# 1. Unstructured Knowledge / Hybrid RAG Models
# ==========================================

class DocumentChunk(Base):
    """
    Stores document chunks with dual representations:
    - Dense vector embedding (384-dimensional for all-MiniLM-L6-v2)
    - Sparse tsvector for PostgreSQL full-text search (BM25 ranking)
    """
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    document_name: Mapped[str] = mapped_column(String(255), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    
    # 384-dimensional dense vector for sentence-transformers/all-MiniLM-L6-v2
    embedding: Mapped[List[float]] = mapped_column(Vector(384), nullable=True)
    
    # Sparse Full-Text Search tsvector for PostgreSQL keyword BM25 ranking
    tsv_content: Mapped[Any] = mapped_column(TSVECTOR, nullable=True)
    
    # Rich metadata: page_number, file_type, token_count, source_url
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=True, default=dict)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        # GIN index for high-speed sparse text search
        Index("ix_document_chunks_tsv", "tsv_content", postgresql_using="gin"),
        # Standard index on document_id and chunk_index
        Index("ix_document_chunks_doc_chunk", "document_id", "chunk_index"),
    )

    def __repr__(self) -> str:
        return f"<DocumentChunk(id={self.id}, doc='{self.document_name}', chunk={self.chunk_index})>"


# ==========================================
# 2. Structured Enterprise Relational Models (Text-to-SQL)
# ==========================================

class Customer(Base):
    """Enterprise Customer / Account record."""
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(120), unique=True, nullable=False, index=True)
    country: Mapped[str] = mapped_column(String(60), nullable=False, default="USA")
    tier: Mapped[str] = mapped_column(String(20), nullable=False, default="Standard")  # Standard, Gold, Enterprise
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    orders: Mapped[List["Order"]] = relationship("Order", back_populates="customer", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Customer(id={self.id}, name='{self.name}', tier='{self.tier}')>"


class Product(Base):
    """Enterprise Product / Catalog item."""
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sku: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    category: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    stock_quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    order_items: Mapped[List["OrderItem"]] = relationship("OrderItem", back_populates="product")

    def __repr__(self) -> str:
        return f"<Product(sku='{self.sku}', name='{self.name}', price={self.price})>"


class Order(Base):
    """Customer Sales Order."""
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[int] = mapped_column(Integer, ForeignKey("customers.id"), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="Completed")  # Pending, Completed, Cancelled
    total_amount: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    order_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    customer: Mapped["Customer"] = relationship("Customer", back_populates="orders")
    items: Mapped[List["OrderItem"]] = relationship("OrderItem", back_populates="order", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Order(id={self.id}, customer_id={self.customer_id}, total={self.total_amount})>"


class OrderItem(Base):
    """Individual line items within a customer order."""
    __tablename__ = "order_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[int] = mapped_column(Integer, ForeignKey("orders.id"), nullable=False, index=True)
    product_id: Mapped[int] = mapped_column(Integer, ForeignKey("products.id"), nullable=False, index=True)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    unit_price: Mapped[float] = mapped_column(Float, nullable=False)

    order: Mapped["Order"] = relationship("Order", back_populates="items")
    product: Mapped["Product"] = relationship("Product", back_populates="order_items")

    def __repr__(self) -> str:
        return f"<OrderItem(order_id={self.order_id}, product_id={self.product_id}, qty={self.quantity})>"
