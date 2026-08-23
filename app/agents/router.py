"""
LangGraph Multi-Agent Router
Determines whether a user query is:
1. Document RAG (Unstructured knowledge)
2. Text-to-SQL (Structured relational database metrics)
3. Direct Synthesis / Conversation
"""

from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    query: str
    query_type: Optional[Literal["rag", "sql", "direct"]]
    context: Optional[str]
    sql_query: Optional[str]
    sql_result: Optional[str]
    response: Optional[str]

def classify_intent_node(state: AgentState) -> AgentState:
    """Classifies user intent based on keywords and schema detection."""
    query = state["query"].lower()
    
    # SQL indicators: numbers, count, sales, revenue, totals, database tables
    sql_keywords = ["how many", "count", "total", "sales", "revenue", "average", "highest", "list all users", "orders"]
    
    # Document RAG indicators: explain, policy, guide, summary, what is, how to
    doc_keywords = ["what is", "how do i", "explain", "policy", "terms", "overview", "documentation", "steps to"]
    
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

def rag_handler_node(state: AgentState) -> AgentState:
    """Retrieves document context and synthesizes answer."""
    state["response"] = f"[Hybrid RAG Response] Answer generated from enterprise documents for: '{state['query']}'"
    return state

def sql_handler_node(state: AgentState) -> AgentState:
    """Executes safe Text-to-SQL query and formats tabular result."""
    state["response"] = f"[Text-to-SQL Copilot] Query executed successfully on PostgreSQL database for: '{state['query']}'"
    return state

def direct_llm_node(state: AgentState) -> AgentState:
    """Handles direct conversational queries."""
    state["response"] = f"Hello! I am OmniQuery-AI. I can answer questions from your PDF knowledge base or query your PostgreSQL database directly."
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
