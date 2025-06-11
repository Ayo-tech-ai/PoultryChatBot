import streamlit as st
import faiss
import pickle
import requests
import json
from sentence_transformers import SentenceTransformer

# ------------------------
# 🔐 Load Hugging Face Token from Secrets
# ------------------------
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
headers = {
    "Authorization": f"Bearer {st.secrets["HF_TOKEN"]}",
    "Content-Type": "application/json"
}

# ------------------------
# 📚 Load Embeddings and Index
# ------------------------
index = faiss.read_index("chikka_index.faiss")
with open("chikka_metadata.pkl", "rb") as f:
    chunks = pickle.load(f)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_top_chunks(query, k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [chunks[i].page_content for i in indices[0]]

# ------------------------
# 🧠 RAG-Based QA with Strict Prompt
# ------------------------
def rag_respond(question, k=3, max_context_chars=1200):
    top_chunks = retrieve_top_chunks(question, k)
    context_text = "\n\n".join(top_chunks).strip()

    if not context_text or len(context_text) < 50:
        return "I'm sorry, I couldn't find any relevant information based on my current knowledge base."

    if len(context_text) > max_context_chars:
        context_text = context_text[:max_context_chars] + "..."

    prompt = f"""
You are Chikka, a friendly but reliable AI assistant for backyard poultry farmers in Nigeria.

ONLY use the information in the 'Context' below to answer.
If you don't find the answer in the context, reply: "I'm sorry, I don't have information about that yet."

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
        return "⚠️ Failed to reach the language model. Please check your token or internet connection."

    result = response.json()
    return result[0]["generated_text"].strip()

# ------------------------
# 💬 Streamlit Interface
# ------------------------
st.set_page_config(page_title="Chikka AI Assistant", layout="centered")
st.title("🐣 Chikka – Your Backyard Poultry Assistant")
st.markdown("Ask me anything about raising broilers, noilers, or cockerels!")

user_question = st.text_input("📩 Type your question here")

if st.button("Ask Chikka"):
    if user_question:
        with st.spinner("Thinking..."):
            answer = rag_respond(user_question)
        st.success("💡 Chikka's Response:")
        st.write(answer)
