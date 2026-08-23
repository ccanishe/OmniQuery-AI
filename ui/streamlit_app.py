import streamlit as st
import httpx
import time

st.set_page_config(
    page_title="OmniQuery-AI Copilot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 OmniQuery-AI: Enterprise Copilot")
st.caption("Hybrid RAG (pgvector + BM25) + LangGraph Agent + Text-to-SQL")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input("FastAPI Backend URL", value="http://localhost:8000")
    st.divider()
    st.markdown("### 📊 Active Engine Capabilities")
    st.success("✅ Hybrid Search (Dense + Sparse BM25)")
    st.success("✅ LangGraph Agentic Router")
    st.success("✅ PostgreSQL Text-to-SQL")
    st.success("✅ RAGAS Automated Evaluation")
    
# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a question about documents or database records..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(f"{api_url}/api/v1/query", json={"query": prompt})
                if res.status_code == 200:
                    data = res.json()
                    route = data.get("route_selected", "direct").upper()
                    response_body = data.get("response", "")
                    
                    # Display route badge
                    st.badge(f"Routing: {route}", icon="🔀")
                    
                    # Simulate streaming for smooth UI feedback
                    for chunk in response_body.split(" "):
                        full_response += chunk + " "
                        time.sleep(0.03)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                else:
                    message_placeholder.error(f"Backend returned error: {res.status_code}")
        except Exception as e:
            message_placeholder.error(f"Could not connect to FastAPI backend at {api_url}. Is it running? (Error: {e})")

    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})
