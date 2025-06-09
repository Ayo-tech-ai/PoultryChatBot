import streamlit as st
import faiss
import pickle
import os
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

# ----------------------------
# Load Embeddings & Metadata
# ----------------------------
embedding_dir = "/content/drive/MyDrive/Chikka AI Assistant/embedding/"

index = faiss.read_index(embedding_dir + "chikka_index.faiss")

with open(embedding_dir + "chikka_metadata.pkl", "rb") as f:
    chunks = pickle.load(f)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# Load TinyLLaMA Model
# ----------------------------
llm = Llama(
    model_path="/content/drive/MyDrive/Chikka_AI_Assistant/llm_models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=2,
    use_mlock=False
)

# ----------------------------
# FAISS Search Function
# ----------------------------
def retrieve_top_chunks(query, k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, k)
    top_chunks = [chunks[i].page_content for i in indices[0]]
    return top_chunks

# ----------------------------
# RAG Function
# ----------------------------
def rag_respond(question, k=3, max_context_tokens=1000):
    top_chunks = retrieve_top_chunks(question, k)
    context_text = "\n\n".join(top_chunks)
    if len(context_text) > max_context_tokens:
        context_text = context_text[:max_context_tokens] + "..."

    prompt = f"""You are an AI assistant for backyard poultry farmers.
Use the information below to answer the question clearly and accurately.

Context:
{context_text}

Question: {question}
Answer:"""

    response = llm(
        prompt,
        max_tokens=400,
        temperature=0.3,
        stop=["\n", "Question:", "Answer:"]
    )
    answer = response['choices'][0]['text'].strip()
    return answer

# ----------------------------
# Streamlit Interface
# ----------------------------
st.set_page_config(page_title="🐣 Chikka AI Assistant", layout="centered")
st.title("🐣 Chikka — Your Backyard Poultry AI Assistant")
st.markdown("Ask me anything about raising broilers, noilers, or cockerels!")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("What would you like to know?", placeholder="e.g. Why are my chicks dying suddenly?")

if st.button("Ask Chikka") and user_input:
    with st.spinner("Chikka is thinking..."):
        reply = rag_respond(user_input)
        st.session_state.history.append((user_input, reply))

# Display chat history
for q, a in reversed(st.session_state.history):
    st.markdown(f"**👤 You:** {q}")
    st.markdown(f"**🐣 Chikka:** {a}")
    st.markdown("---")
