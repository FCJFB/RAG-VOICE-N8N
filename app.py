import os
import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1/query")

st.set_page_config(
    page_title="CachyOS Local RAG Assistant",
    page_icon="🐧",
    layout="centered"
)

st.title("🐧 CachyOS Local RAG Assistant")
st.caption("Privacy-first documentation assistant powered by Llama 3.2, ChromaDB, and CoT reasoning.")

# Initialize chat session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "reasoning_plan" in msg and msg["reasoning_plan"]:
            with st.expander("🧠 View LLM Reasoning Plan (Pass 1)"):
                st.code(msg["reasoning_plan"], language="markdown")
        if "distance" in msg and msg["distance"] is not None:
            st.caption(f"Vector Distance Score: `{msg['distance']}`")

# Process new user prompt
if user_prompt := st.chat_input("Ask a question about CachyOS..."):
    # Render user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Call FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking & searching documentation..."):
            try:
                res = requests.post(API_URL, json={"query": user_prompt}, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    response_text = data["response"]
                    reasoning_plan = data.get("reasoning_plan")
                    distance = data.get("vector_distance")

                    # Display final answer
                    st.markdown(response_text)

                    # Display Pass 1 reasoning accordion if present
                    if reasoning_plan:
                        with st.expander("🧠 View LLM Reasoning Plan (Pass 1)"):
                            st.code(reasoning_plan, language="markdown")

                    if distance is not None:
                        st.caption(f"Vector Distance Score: `{distance}`")

                    # Save to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "reasoning_plan": reasoning_plan,
                        "distance": distance
                    })
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to FastAPI server. Ensure `uvicorn src.api:app` is running on port 8000.")