"""
Unit & Integration Tests for Week 2 Text-to-SQL Copilot Engine
Tests:
1. SQL Sanitization & Validation (Valid SELECT statements)
2. Security Sandbox blocking malicious mutations (DROP, DELETE, UPDATE, stacked statements)
3. Safety LIMIT appending
4. Markdown table formatting
5. End-to-end Text-to-SQL pipeline execution
"""

import pytest
from app.agents.sql_agent import (
    validate_and_sanitize_sql,
    format_sql_results_as_markdown,
    run_text_to_sql_pipeline
)


def test_sql_validation_safe_queries():
    """Verifies that clean SELECT queries pass validation."""
    query = "SELECT id, name, price FROM products WHERE price > 100"
    sanitized = validate_and_sanitize_sql(query)
    assert sanitized.startswith("SELECT")
    assert "LIMIT 50" in sanitized


def test_sql_validation_blocks_drop_table():
    """Verifies security sandbox blocks DROP TABLE injection."""
    malicious = "DROP TABLE products;"
    with pytest.raises(ValueError, match="Security Violation"):
        validate_and_sanitize_sql(malicious)


def test_sql_validation_blocks_delete_mutation():
    """Verifies security sandbox blocks DELETE FROM mutations."""
    malicious = "DELETE FROM orders WHERE id = 1"
    with pytest.raises(ValueError, match="Security Violation"):
        validate_and_sanitize_sql(malicious)


def test_sql_validation_blocks_multiple_stacked_statements():
    """Verifies security sandbox blocks stacked query injection."""
    stacked = "SELECT * FROM customers; DROP TABLE orders;"
    with pytest.raises(ValueError, match="Security Violation"):
        validate_and_sanitize_sql(stacked)


def test_markdown_table_formatting():
    """Verifies tabular results format into clean GitHub Markdown tables."""
    cols = ["sku", "name", "price"]
    rows = [
        {"sku": "SKU-SRV-101", "name": "Enterprise Server 1U", "price": 2499.00},
        {"sku": "SKU-SW-202", "name": "AI Analytics Pro", "price": 499.00}
    ]
    markdown = format_sql_results_as_markdown(
        user_query="List high value products",
        sql_query="SELECT sku, name, price FROM products LIMIT 2",
        columns=cols,
        rows=rows
    )
    assert "| sku | name | price |" in markdown
    assert "| SKU-SRV-101 | Enterprise Server 1U | 2499.0 |" in markdown
    assert "2 records" in markdown


@pytest.mark.asyncio
async def test_end_to_end_text_to_sql_pipeline():
    """Verifies live query execution against the seeded database."""
    query = "How many total orders are in Completed status?"
    sql, response, summary = await run_text_to_sql_pipeline(query)
    
    assert sql != "BLOCKED"
    assert "SELECT" in sql.upper()
    assert "Text-to-SQL Copilot Engine" in response
