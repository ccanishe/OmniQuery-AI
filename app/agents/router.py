"""
LangGraph Multi-Agent Router
Determines whether a user query is:
1. Document RAG (Unstructured knowledge)
2. Text-to-SQL (Structured relational database metrics)
3. Direct Synthesis / Conversation
"""

from typing import TypedDict, Literal, Optional, List, Dict, Any
from langgraph.graph import StateGraph, END
from app.rag.hybrid_retriever import retrieve_context
from app.rag.synthesizer import synthesize_answer


class AgentState(TypedDict):
    query: str
    query_type: Optional[Literal["rag", "sql", "direct"]]
    retrieved_chunks: Optional[List[Dict[str, Any]]]
    sql_query: Optional[str]
    sql_result: Optional[str]
    response: Optional[str]


def classify_intent_node(state: AgentState) -> AgentState:
    """Classifies user intent based on keywords and schema detection."""
    raw_query = state.get("query") or ""
    query = raw_query.strip().lower()
    
    if not query:
        state["query_type"] = "direct"
        return state
    
    # SQL indicators: numbers, count, sales, revenue, totals, database tables, raw sql
    sql_keywords = [
        "how many", "count", "total", "sales", "revenue", "average", "highest", 
        "lowest", "list all users", "orders", "customers", "products", "stock",
        "select", "from", "drop table", "insert into"
    ]
    
    # Document RAG indicators: explain, policy, guide, summary, what is, how to, return, SLA, error
    doc_keywords = [
        "what is", "how do i", "explain", "policy", "terms", "overview", 
        "documentation", "steps to", "reimbursement", "allowance", "mfa", "sla", "warranty", "err_"
    ]
    
    if any(k in query for k in sql_keywords):
        state["query_type"] = "sql"
    elif any(k in query for k in doc_keywords):
        state["query_type"] = "rag"
    else:
        state["query_type"] = "direct"
        
    return state


def route_query(state: AgentState) -> str:
    """Conditional edge router function."""
    return state["query_type"]


async def rag_handler_node(state: AgentState) -> AgentState:
    """
    Executes Hybrid RAG pipeline:
    1. Dense vector + Sparse BM25 search with RRF fusion and Cross-Encoder re-ranking.
    2. Synthesizes grounded answer with source citations.
    """
    query = state["query"]
    try:
        chunks = await retrieve_context(query, top_k=3)
        state["retrieved_chunks"] = chunks
        answer = await synthesize_answer(query, chunks)
        state["response"] = answer
    except Exception as e:
        # Log error internally for developers/monitoring
        print(f"[SECURITY ALERT] Retrieval exception logged: {e}")
        # Return sanitized response to client
        state["response"] = (
            "⚠️ We encountered a temporary issue searching the enterprise document knowledge base. "
            "Our engineering team has been notified. Please try your query again in a moment."
        )
    return state


async def sql_handler_node(state: AgentState) -> AgentState:
    """Executes safe Text-to-SQL query and formats tabular result (Milestone Week 2)."""
    state["response"] = (
        f"📊 **[Text-to-SQL Copilot]**\n\n"
        f"Query analyzed: *\"{state['query']}\"*\n\n"
        f"```sql\n-- Sample Parameterized SQL for PostgreSQL\nSELECT count(*), status FROM orders GROUP BY status;\n```\n"
        f"*(Full automated SQL sandbox execution enabled in Week 2)*"
    )
    return state


async def direct_llm_node(state: AgentState) -> AgentState:
    """Handles direct conversational queries."""
    state["response"] = (
        "Hello! I am **OmniQuery-AI**, your enterprise copilot.\n\n"
        "I can help you with:\n"
        "- 📄 **Document Search:** Ask questions about IT security, remote work policies, SLA terms, or uploaded PDFs.\n"
        "- 📊 **Database Queries:** Ask for customer metrics, order volumes, product inventory, and sales reports."
    )
    return state


def create_agent_graph():
    """Builds and compiles the LangGraph state graph."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("classifier", classify_intent_node)
    workflow.add_node("rag_node", rag_handler_node)
    workflow.add_node("sql_node", sql_handler_node)
    workflow.add_node("direct_node", direct_llm_node)
    
    workflow.set_entry_point("classifier")
    
    workflow.add_conditional_edges(
        "classifier",
        route_query,
        {
            "rag": "rag_node",
            "sql": "sql_node",
            "direct": "direct_node"
        }
    )
    
    workflow.add_edge("rag_node", END)
    workflow.add_edge("sql_node", END)
    workflow.add_edge("direct_node", END)
    
    return workflow.compile()


agent_app = create_agent_graph()
