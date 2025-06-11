import streamlit as st
import faiss
import pickle
import requests
import json
from sentence_transformers import SentenceTransformer

# --- Load FAISS index, metadata, and embedding model ---
index = faiss.read_index("chikka_index.faiss")

with open("chikka_metadata.pkl", "rb") as f:
    chunks = pickle.load(f)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# --- FAISS Retrieval Function ---
def retrieve_top_chunks(query, k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [chunks[i].page_content for i in indices[0]]


# --- Hugging Face API Setup ---
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

headers = {
    "Authorization": f"Bearer {st.secrets['HUGGINGFACE_API_TOKEN']}",
    "Content-Type": "application/json"
}


# --- RAG-Powered QA Function ---
def rag_respond(question, k=3, max_context_chars=1200):
    top_chunks = retrieve_top_chunks(question, k)
    context_text = "\n\n".join(top_chunks).strip()

    if not context_text or len(context_text) < 50:
        return "I'm sorry, I couldn't find any relevant information based on my current knowledge base."

    if len(context_text) > max_context_chars:
        context_text = context_text[:max_context_chars] + "..."

    prompt = f"""
You are Chikka, a friendly but strict AI assistant for backyard poultry farmers in Nigeria.

ONLY use the information provided in the 'Context' section to answer the question.
If you do not find the answer in the context, respond with:
"I'm sorry, I don't have information about that yet."

Context:
{context_text}

Question: {question}
Answer:
"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.0,
            "max_new_tokens": 400,
            "stop": ["\nQuestion:", "Answer:", "Context:"]
        }
    }

    response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
    
    if response.status_code != 200:
        st.error("⚠️ Failed to reach the language model. Please check your token or internet connection.")
        return ""

    result = response.json()

    try:
        return result[0]["generated_text"].split("Answer:")[-1].strip()
    except Exception:
        return "⚠️ Unexpected response from model."


# --- Streamlit Interface ---
st.set_page_config(page_title="Chikka AI Assistant", layout="centered")
st.title("🐣 Chikka – Your Backyard Poultry Assistant")
st.markdown("Ask me anything about raising broilers, noilers, or cockerels!")

user_question = st.text_input("📩 Type your question here")

if st.button("Ask Chikka"):
    if user_question:
        with st.spinner("💬 Chikka is thinking..."):
            answer = rag_respond(user_question)
        st.success("💡 Chikka's Response:")
        st.write(answer)
