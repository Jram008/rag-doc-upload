# import streamlit as st
# import requests

# # FastAPI backend URL
# API_URL = "http://localhost:8000/ask"   # adjust if running on another host/port

# st.set_page_config(page_title="FAISS RAG Chatbot", page_icon="🤖")

# st.title("📚 FAISS RAG Chatbot")
# st.write("Ask questions based on the knowledge base (PDF indexed with FAISS).")

# # Session state for chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "session_id" not in st.session_state:
#     st.session_state.session_id = "user1"   # could generate unique IDs per user

# # Display chat history
# for msg in st.session_state.messages:
#     if msg["role"] == "user":
#         st.chat_message("user").markdown(msg["content"])
#     else:
#         st.chat_message("assistant").markdown(msg["content"])

# # Input box
# if prompt := st.chat_input("Type your question..."):
#     # Add user message to history
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     st.chat_message("user").markdown(prompt)

#     # Call FastAPI backend
#     try:
#         response = requests.post(API_URL, json={
#             "query": prompt,
#             "session_id": st.session_state.session_id
#         })
#         response.raise_for_status()
#         answer = response.json()["answer"]
#     except Exception as e:
#         answer = f"⚠️ Error: {e}"

#     # Add assistant message to history
#     st.session_state.messages.append({"role": "assistant", "content": answer})
#     st.chat_message("assistant").markdown(answer)

import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="FAISS RAG Chatbot", page_icon="🤖")
st.title("📚 FAISS RAG Chatbot")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF to index", type=["pdf"])
if uploaded_file is not None:
    st.write("Uploading and indexing...")
    files = {"file": uploaded_file.getvalue()}
    try:
        res = requests.post(f"{API_URL}/upload_pdf", files={"file": uploaded_file})
        st.success(res.json()["message"])
    except Exception as e:
        st.error(f"Upload failed: {e}")

# Session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "user1"

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Type your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    try:
        response = requests.post(f"{API_URL}/ask", json={
            "query": prompt,
            "session_id": st.session_state.session_id
        })
        response.raise_for_status()
        answer = response.json()["answer"]
    except Exception as e:
        answer = f"⚠️ Error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").markdown(answer)
