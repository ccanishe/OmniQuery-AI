"""
LLM Response Synthesizer Module
Grounds LLM generation in retrieved document context chunks with strict
anti-hallucination guardrails and source citations.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

SYSTEM_GROUNDING_PROMPT = """
You are OmniQuery-AI, an enterprise intelligence assistant.
Your task is to answer the user's question accurately and concisely using ONLY the provided document context passages.

RULES:
1. Rely strictly on the facts stated in the context. Do NOT extrapolate or assume information not present.
2. If the provided context does not contain enough information to answer the question, explicitly respond: "I cannot find sufficient information in the enterprise documents to answer this question."
3. Always cite your sources at the end of the response using the format:
   - Sources: [Document Name] (Chunk # / Page #)
4. Keep the tone professional, direct, and structured with bullet points where appropriate.
""".strip()


def format_context_passages(chunks: List[Dict[str, Any]]) -> str:
    """Formats retrieved chunks into a clean context string for prompt injection."""
    if not chunks:
        return "No relevant documents found."

    formatted_parts = []
    for idx, chunk in enumerate(chunks):
        doc_name = chunk.get("document_name", "Unknown")
        meta = chunk.get("metadata_json", {})
        page = meta.get("page_number", "N/A")
        content = chunk.get("content", "").strip()
        formatted_parts.append(
            f"--- [Passage {idx + 1}] Document: {doc_name} (Page: {page}) ---\n{content}"
        )
    return "\n\n".join(formatted_parts)


async def synthesize_answer(query: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Generates a grounded response using Gemini API, Ollama, or fallback generator.
    """
    context_str = format_context_passages(chunks)

    # Check for Gemini API key
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"{SYSTEM_GROUNDING_PROMPT}\n\nCONTEXT:\n{context_str}\n\nQUESTION: {query}\n\nANSWER:"
            response = await model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API generation error: {e}")

    # Fallback to structured extractive answer if LLM key is not configured locally
    if not chunks:
        return "I could not find any relevant information in the enterprise knowledge base for your query."

    top_chunk = chunks[0]
    doc_name = top_chunk.get("document_name", "Enterprise Document")
    meta = top_chunk.get("metadata_json", {})
    page = meta.get("page_number", 1)

    return (
        f"Based on the enterprise document **{doc_name}** (Page {page}):\n\n"
        f"> {top_chunk.get('content', '').strip()}\n\n"
        f"**Sources:**\n"
        f"- {doc_name} (Chunk #{top_chunk.get('chunk_index', 1)}, Page {page})"
    )
