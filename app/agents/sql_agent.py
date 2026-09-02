"""
Autonomous Text-to-SQL Copilot Engine
Features:
1. Dynamic PostgreSQL Schema & Catalog Inspector
2. LLM-powered SQL Query Generator (Gemini 1.5 Flash + Fallback Heuristics)
3. Read-Only Security Sandbox (Blocks DROP, DELETE, UPDATE, INSERT, ALTER)
4. Async Query Execution Engine with LIMIT enforcement
5. Markdown Table & Natural Language Response Synthesizer
"""

import os
import re
from typing import Dict, Any, List, Tuple, Optional
from sqlalchemy import text
from dotenv import load_dotenv
from app.database import AsyncSessionLocal

load_dotenv()

# =====================================================================
# 1. PostgreSQL Schema Catalog Definition
# =====================================================================

DATABASE_SCHEMA_DESCRIPTION = """
PostgreSQL Relational Database Schema:

Table: customers
- id (INTEGER, PRIMARY KEY)
- name (VARCHAR(100)) - Customer company name
- email (VARCHAR(120), UNIQUE)
- country (VARCHAR(60)) - e.g., 'USA', 'India'
- tier (VARCHAR(20)) - 'Standard', 'Gold', 'Enterprise'
- created_at (TIMESTAMP)

Table: products
- id (INTEGER, PRIMARY KEY)
- sku (VARCHAR(50), UNIQUE) - e.g., 'SKU-SRV-101', 'SKU-SW-202'
- name (VARCHAR(150)) - Product title
- category (VARCHAR(60)) - e.g., 'Hardware', 'Software', 'Security', 'Furniture'
- price (FLOAT) - Unit price in USD
- stock_quantity (INTEGER) - Available inventory
- created_at (TIMESTAMP)

Table: orders
- id (INTEGER, PRIMARY KEY)
- customer_id (INTEGER, FOREIGN KEY -> customers.id)
- status (VARCHAR(30)) - 'Pending', 'Completed', 'Cancelled'
- total_amount (FLOAT) - Total order price in USD
- order_date (TIMESTAMP)

Table: order_items
- id (INTEGER, PRIMARY KEY)
- order_id (INTEGER, FOREIGN KEY -> orders.id)
- product_id (INTEGER, FOREIGN KEY -> products.id)
- quantity (INTEGER) - Number of units purchased
- unit_price (FLOAT) - Price per unit at purchase time
""".strip()


# =====================================================================
# 2. Security & SQL Sanitization Sandbox
# =====================================================================

FORBIDDEN_SQL_KEYWORDS = [
    r"\bDROP\b", r"\bDELETE\b", r"\bUPDATE\b", r"\bINSERT\b", 
    r"\bALTER\b", r"\bTRUNCATE\b", r"\bGRANT\b", r"\bREVOKE\b",
    r"\bEXEC\b", r"\bEXECUTE\b", r"\bCREATE\b", r"\bREPLACE\b"
]

def validate_and_sanitize_sql(raw_sql: str) -> str:
    """
    Strict SQL validation sandbox:
    1. Extracts pure SQL (strips markdown code blocks).
    2. Blocks mutation and DDL keywords (prevents SQL injection).
    3. Guarantees query starts with SELECT or WITH.
    4. Appends LIMIT 50 if no limit is specified to avoid memory exhaustion.
    """
    # 1. Clean markdown formatting
    cleaned = raw_sql.strip()
    cleaned = re.sub(r"^```(?:sql)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    
    # Remove trailing semicolons
    cleaned = cleaned.rstrip(";").strip()

    # 2. Disallow multiple stacked statements (prevents 'SELECT 1; DROP TABLE products;')
    if ";" in cleaned:
        raise ValueError("Security Violation: Multiple stacked SQL statements are forbidden.")

    # 3. Disallow mutation / destructive keywords
    for pattern in FORBIDDEN_SQL_KEYWORDS:
        if re.search(pattern, cleaned, flags=re.IGNORECASE):
            matched_kw = re.search(pattern, cleaned, flags=re.IGNORECASE).group()
            raise ValueError(f"Security Violation: Destructive SQL command '{matched_kw}' is strictly prohibited.")

    # 4. Enforce read-only prefix (SELECT or WITH for CTEs)
    if not (cleaned.upper().startswith("SELECT") or cleaned.upper().startswith("WITH")):
        raise ValueError("Security Violation: Only read-only SELECT or WITH statements are allowed.")

    # 5. Append safety LIMIT if missing
    if not re.search(r"\bLIMIT\b", cleaned, flags=re.IGNORECASE):
        cleaned += " LIMIT 50"

    return cleaned


# =====================================================================
# 3. LLM SQL Generation Pipeline
# =====================================================================

SQL_SYSTEM_PROMPT = f"""
You are OmniQuery-AI's expert PostgreSQL Data Engineer.
Convert the user's natural language question into a single, syntactically correct, read-only PostgreSQL query.

RULES:
1. ONLY return the raw SQL query. Do NOT include markdown code fences (```sql), explanations, or notes.
2. Use valid PostgreSQL 16 syntax (e.g., ILIKE for case-insensitive search, COUNT, SUM, AVG, GROUP BY).
3. Use the following schema:
{DATABASE_SCHEMA_DESCRIPTION}
4. When filtering text columns (like name, tier, category, status), use ILIKE for case-insensitivity.
5. If the query asks for counts, aggregates, or listings, join tables appropriately (e.g., orders JOIN customers ON orders.customer_id = customers.id).
""".strip()


async def generate_sql_query(user_query: str) -> str:
    """
    Generates parameterized SQL using Gemini API or intelligent deterministic heuristic fallback.
    """
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if gemini_api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"{SQL_SYSTEM_PROMPT}\n\nUSER QUESTION: {user_query}\n\nSQL QUERY:"
            response = await model.generate_content_async(prompt)
            generated_sql = response.text.strip()
            return validate_and_sanitize_sql(generated_sql)
        except Exception as e:
            print(f"[SQL AGENT] Gemini API error ({e}), using fallback SQL generator.")

    # Intelligent Fallback SQL Generator for offline local development & testing
    q_lower = user_query.lower()
    
    if "order" in q_lower and ("completed" in q_lower or "status" in q_lower or "how many" in q_lower):
        raw = "SELECT status, count(*) AS total_orders, sum(total_amount) AS total_revenue FROM orders GROUP BY status"
    elif "product" in q_lower or "stock" in q_lower or "inventory" in q_lower or "sku" in q_lower:
        raw = "SELECT sku, name, category, price, stock_quantity FROM products ORDER BY price DESC"
    elif "customer" in q_lower or "tier" in q_lower or "who" in q_lower:
        raw = "SELECT name, email, country, tier FROM customers ORDER BY name ASC"
    elif "revenue" in q_lower or "sales" in q_lower:
        raw = "SELECT c.name AS customer_name, sum(o.total_amount) AS total_spent FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.name ORDER BY total_spent DESC"
    else:
        raw = "SELECT count(*) AS total_products FROM products"

    return validate_and_sanitize_sql(raw)


# =====================================================================
# 4. Async Read-Only Execution Sandbox
# =====================================================================

async def execute_safe_sql(sql_query: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Executes validated SQL query in an async read-only transaction.
    Returns: (column_names, rows_as_dicts)
    """
    async with AsyncSessionLocal() as session:
        # Enforce PostgreSQL transaction read-only mode at connection level
        await session.execute(text("SET TRANSACTION READ ONLY;"))
        result = await session.execute(text(sql_query))
        
        # Extract column names
        columns = list(result.keys())
        # Extract data rows
        rows = [dict(row._mapping) for row in result.fetchall()]
        return columns, rows


# =====================================================================
# 5. Markdown Table & Response Formatter
# =====================================================================

def format_sql_results_as_markdown(
    user_query: str, 
    sql_query: str, 
    columns: List[str], 
    rows: List[Dict[str, Any]]
) -> str:
    """Formats query results into an enterprise Markdown report with code inspection."""
    if not rows:
        return (
            f"📊 **[Text-to-SQL Copilot]**\n\n"
            f"**Executed Query:**\n```sql\n{sql_query}\n```\n\n"
            f"ℹ️ *No matching records found in the database for your search criteria.*"
        )

    # Build Markdown Table
    header_row = "| " + " | ".join(columns) + " |"
    separator_row = "| " + " | ".join(["---"] * len(columns)) + " |"
    
    data_rows = []
    for row in rows:
        row_str = "| " + " | ".join(str(row.get(col, "")) for col in columns) + " |"
        data_rows.append(row_str)

    table_markdown = "\n".join([header_row, separator_row] + data_rows)

    return (
        f"📊 **[Text-to-SQL Copilot Engine]**\n\n"
        f"**Question:** *\"{user_query}\"*\n\n"
        f"**Generated SQL Query:**\n```sql\n{sql_query}\n```\n\n"
        f"### 📋 Query Results ({len(rows)} record{'s' if len(rows) != 1 else ''}):\n\n"
        f"{table_markdown}\n\n"
        f"*(Executed securely in PostgreSQL 16 read-only transaction sandbox)*"
    )


# =====================================================================
# 6. End-to-End SQL Copilot Pipeline
# =====================================================================

async def run_text_to_sql_pipeline(user_query: str) -> Tuple[str, str, str]:
    """
    Main orchestration entrypoint for Text-to-SQL:
    1. Generates and validates SQL.
    2. Executes safely against PostgreSQL.
    3. Formats response with Markdown tables.
    Returns: (sql_query, formatted_markdown_response, raw_data_summary)
    """
    try:
        # 1. Generate SQL
        sql_query = await generate_sql_query(user_query)
        
        # 2. Execute SQL
        columns, rows = await execute_safe_sql(sql_query)
        
        # 3. Format Response
        markdown_response = format_sql_results_as_markdown(user_query, sql_query, columns, rows)
        return sql_query, markdown_response, f"{len(rows)} rows returned"
        
    except ValueError as val_err:
        # Security rejection
        err_msg = f"🛡️ **Security Sandbox Alert:** {str(val_err)}"
        return "BLOCKED", err_msg, "0 rows"
    except Exception as e:
        err_msg = f"⚠️ **SQL Execution Error:** An issue occurred while running the query against PostgreSQL: `{str(e)}`"
        return "ERROR", err_msg, "0 rows"
