import os

import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/v1/query")

st.set_page_config(page_title="Academic Lecture Assistant", layout="wide")
st.title("📚 Lecture RAG Assistant")

# Sidebar Filters
st.sidebar.header("Filter Context")
course_id = st.sidebar.text_input("Course ID (optional)", value="IMAGEPROCESSING")
lecture_num = st.sidebar.number_input("Lecture Number (optional)", min_value=0, value=0, step=1)

lecture_filter = lecture_num if lecture_num > 0 else None
course_filter = course_id.strip() if course_id.strip() else None

query = st.text_input("Ask a question about your lecture slides:")

if st.button("Submit Query") and query:
    payload = {
        "query": query,
        "course_id": course_filter,
        "lecture_num": lecture_filter
    }
    
    with st.spinner("Analyzing lecture slides..."):
        try:
            res = requests.post(API_URL, json=payload)
            if res.status_code == 200:
                data = res.json()
                st.subheader("Response")
                st.write(data["response"])
                
                if data["is_relevant"]:
                    st.info(f"Vector Distance Score: {data['vector_distance']}")
                    with st.expander("View Sources"):
                        for src in data.get("sources", []):
                            st.write(f"• **File:** {src['file']} | **Slide:** {src['slide']} | **Course:** {src['course']}")
                    with st.expander("View Reasoning Plan"):
                        st.write(data["reasoning_plan"])
            else:
                st.error(f"Error {res.status_code}: {res.text}")
        except Exception as e:
            st.error(f"Failed to reach API server: {e}")