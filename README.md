# RAGModel 🔍📚🤖  
_Retrieval-Augmented Generation for Intelligent Context-Aware Answering_

![Langchain](https://img.shields.io/badge/LangChain-Compatible-blueviolet)
![RAG](https://img.shields.io/badge/Powered%20By-Retrieval%20Augmented%20Generation-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)

---

## 🧠 Overview

**RAGModel** implements a **Retrieval-Augmented Generation (RAG)** pipeline that combines the power of large language models (LLMs) with external knowledge retrieval. This allows the system to generate accurate and contextually relevant answers, even for domain-specific queries not directly stored in the model.

The system retrieves relevant documents from a vector store using semantic search and passes them as context to an LLM for final response generation.

---

## 🔧 Components

- 📄 **Document Loader**: Ingests and parses PDF/text/HTML files into chunks.
- 🧠 **Embeddings**: Converts text chunks into dense vector representations (e.g., OpenAI/BERT/Gemini).
- 📦 **Vector Store**: Stores embeddings and supports similarity search (e.g., Pinecone, FAISS).
- 🔍 **Retriever**: Finds top-k relevant documents from the store for a query.
- 🗨️ **LLM Generator**: Uses an LLM to generate final response based on retrieved context.

---

## 🛠️ Tech Stack

| Component        | Tool/Library         |
|------------------|----------------------|
| Backend Language | Python 3.8+          |
| LLM Integration  |Ollama                | 
| Embeddings       | Sentence Transformers|
| Vector Store     | Pinecone / FAISS     |
| Framework        | LangChain            |
| Document Support | PDFs, TXT, Webpages  |

---

## 🚀 Usage

### Download Ollama
Go to https://ollama.com/ and download the application for your OS (macOS, Windows, or Linux).
Install it.
Open your terminal or command prompt and pull a powerful but efficient model. We'll use a quantized version of Llama 3 8B, which runs well on most modern computers.

```bash
ollama pull llama3:8b-instruct-q4_K_M
```

### Clone the Repository

```bash
git clone https://github.com/Man-asvi/RAGModel.git
cd RAGModel
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Setup .env File

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_environment
INDEX_NAME=your_index_name
```

### Run Code

```bash
python test.py
```

Modify test.py as needed to:
- Load documents
- Generate embeddings
- Store vectors
- Run queries
- Generate answers
