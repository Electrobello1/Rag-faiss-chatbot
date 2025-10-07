# 📚 RAG Chatbot with LangChain and FAISS

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** powered by **LangChain**, **FAISS vector store**, and **OpenAI embeddings**. The chatbot retrieves the most relevant pieces of context from a knowledge base and answers user queries strictly based on that information.  

If the answer is not present in the knowledge base, the bot responds with:  
👉 *“I don’t know based on the available information.”*

---

## 🚀 Features
- ✅ Retrieval-Augmented Generation (RAG) pipeline  
- ✅ FAISS vector store for fast similarity search  
- ✅ OpenAI embeddings for high-quality text representation  
- ✅ Configurable **chunk size** and **similarity threshold**  
- ✅ Strict grounding to context (no hallucinations)  
- ✅ Modular design separating chatbot logic from UI  
- ✅ Intel UHD GPU support using  Openvino
---

## 📂 Project Structure
rag-chatbot/
│── data/ # Knowledge base text files
│── vectorstore/ # Saved FAISS index
│── rag_bot.py # RAG chatbot logic
│── requirements.txt # Dependencies
│── README.md # Project documentation

---

## ⚙️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR-USERNAME/rag-chatbot.git
cd rag-chatbot
```
2.**Create and activate a virtual environment:**
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```
---

3.** Install dependencies
```
pip install -r requirements.txt
```
##  ▶️ Usage
```
python rag_bot.py
```
## ⚙️ Configuration

You can tune parameters directly in rag_bot.py:
```
SIMILARITY_THRESHOLD = 0.5   # Higher = stricter relevance
CHUNK_SIZE = 1000            # Size of text chunks
CHUNK_OVERLAP = 100          # Overlap between chunks
MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
```
## 📜 License
___
This project is licensed under the MIT License.
You are free to use, modify, and distribute it.
___
## 💡 Future Improvements

 Add streaming responses

 Support multiple knowledge bases

 Deploy as a web app with FastAPI

 Add Docker support
 
## 👨‍💻 Author

Developed by Princewill Bello 🚀