# 🐥 Chikka – Your Friendly Poultry AI Assistant

**Chikka** is an AI-powered extension assistant designed to support backyard poultry farmers in Nigeria. Focused on meat-producing birds like **broilers, noilers, and cockerels**, Chikka provides real-time, accurate, and context-aware responses to common poultry farming queries. Whether it's feeding, disease control, or brooding tips—Chikka is your always-available agri-extension buddy!

---

## 🤖 Project Overview

This project explores how natural language processing (NLP), vector databases, and open-source language models can be applied in agriculture. Chikka uses **Retrieval-Augmented Generation (RAG)** via **FAISS indexing** and a **lightweight LLM** backend to ensure that every answer is fast, informed, and farmer-friendly.

---

## ✨ Features

- 💬 Conversational AI tailored to broiler, noiler & cockerel farmers  
- 🔍 Retrieves accurate info from a FAISS-indexed poultry knowledge base  
- 📚 Localized content based on Nigerian backyard poultry practices  
- 🧠 Supports context-aware multi-turn chat experience  
- 🚀 Powered by open-source LLMs (e.g., Mistral-7B or LLaMA-2)  
- 💻 Optimized for mobile-friendly deployment via Streamlit or Telegram  

---

## 🛠 Tech Stack

| Component              | Technology                              |
|------------------------|------------------------------------------|
| Frontend               | Streamlit                               |
| Backend Language Model | Mistral-7B / LLaMA-2 via Hugging Face Transformers |
| Semantic Search        | FAISS                                   |
| Embedding Model        | Sentence Transformers (`all-MiniLM-L6-v2`) |
| File Support           | `.txt`, `.pdf`, `.docx`                  |

---

## 📁 Project Structure

```bash
Chikka/
├── data/               # Poultry content (disease, feeding, management)
├── embeddings/         # FAISS vector index and metadata
├── app.py              # Main Streamlit chatbot interface
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── utils/              # Text loaders, embedding pipeline, prompt templates


##💡 Use Case

Chikka helps

🐔 Small-scale farmers raising broilers and cockerels in urban/rural backyards

🧑‍🔬 Agri-extension workers who need on-the-go poultry management facts

🏢 Agri-tech startups building farmer support chatbots

👩‍💻 AI researchers exploring NLP for rural extension use



---

🚀 How to Run Locally

1. Clone this repo:



git clone https://github.com/your-username/chikka-poultry-assistant
cd chikka-poultry-assistant

2. Install dependencies:



pip install -r requirements.txt

3. Start the app:



streamlit run app.py

4. Upload your poultry knowledge files and start chatting!




--

🌍 Deployment

This app is deployable on Streamlit Cloud or can be extended to Telegram for mobile-first users.

> 🌐 Live Demo: https://smartagric4ai.streamlit.app




---

👤 Creator

Ayoola Mujib Ayodele
Agri-Extensionist | Data Scientist
📧 ayodelemujibayoola@gmail.com
🔗 LinkedIn
🔗 GitHub


---

🤝 Contributing

Contributions are welcome!
Feel free to open issues, suggest improvements, or fork the repo.


---

📜 License

This project is currently unlicensed.
For reuse, contributions, or collaboration, please contact the creator.
