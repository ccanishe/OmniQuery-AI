import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
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
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=1000, 
        description="The user prompt or question"
    )
    user_id: Optional[str] = Field("default_user", max_length=100)

    @field_validator("query")
    @classmethod
    def validate_query_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query string cannot be empty or pure whitespace.")
        return stripped

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
    try:
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
    except Exception as e:
        print(f"[SECURITY ALERT] Unhandled endpoint exception: {e}")
        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred. Please try again later."
        )

@app.post("/api/v1/stream")
async def stream_query(request: QueryRequest):
    """Server-Sent Events (SSE) streaming endpoint for real-time token delivery."""
    async def token_generator():
        try:
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
        except Exception as e:
            print(f"[SECURITY ALERT] Streaming exception: {e}")
            yield "event: error\ndata: ⚠️ An error occurred while generating your response. Please try again.\n\n"
            
    return StreamingResponse(token_generator(), media_type="text/event-stream")

