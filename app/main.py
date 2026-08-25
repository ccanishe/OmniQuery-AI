import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from app.database import init_db
from app.agents.router import agent_app

app = FastAPI(
    title="OmniQuery-AI API",
    description="Enterprise Hybrid RAG & Autonomous SQL Copilot Backend",
    version="1.0.0"
)

@app.on_event("startup")
async def on_startup():
    try:
        await init_db()
        print("Database & pgvector extension initialized successfully.")
    except Exception as e:
        print(f"Notice: Database connection offline or initializing: {e}")

class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = "default_user"

class QueryResponse(BaseModel):
    query: str
    route_selected: str
    response: str

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "OmniQuery-AI",
        "features": ["Hybrid RAG (pgvector + BM25)", "LangGraph Agent", "Text-to-SQL"]
    }

@app.post("/api/v1/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """Synchronous JSON endpoint returning full response."""
    initial_state = {
        "query": request.query,
        "query_type": None,
        "retrieved_chunks": None,
        "sql_query": None,
        "sql_result": None,
        "response": None
    }
    
    result = await agent_app.ainvoke(initial_state)
    return QueryResponse(
        query=request.query,
        route_selected=result.get("query_type", "direct"),
        response=result.get("response", "No response generated.")
    )

@app.post("/api/v1/stream")
async def stream_query(request: QueryRequest):
    """Server-Sent Events (SSE) streaming endpoint for real-time token delivery."""
    async def token_generator():
        initial_state = {
            "query": request.query,
            "query_type": None,
            "retrieved_chunks": None,
            "sql_query": None,
            "sql_result": None,
            "response": None
        }
        
        result = await agent_app.ainvoke(initial_state)
        response_text = result.get("response", "")

        
        # Stream response chunk-by-chunk
        words = response_text.split(" ")
        for word in words:
            yield f"data: {word} \n\n"
            await asyncio.sleep(0.05)
            
    return StreamingResponse(token_generator(), media_type="text/event-stream")
