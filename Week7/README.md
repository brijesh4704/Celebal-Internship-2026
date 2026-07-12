# RAG-Based Document Question Answering System 🤖📄

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system that answers questions from custom PDF documents.

Users can upload a PDF, ask questions about its contents, and receive accurate answers based only on the uploaded document. The application retrieves the most relevant text using **FAISS** and **Sentence Transformers**, then generates answers using **Cohere**.

---

## Features ✨

* 📄 Upload custom PDF documents
* 📚 Extract text from PDFs using PyMuPDF
* ✂️ Split documents into semantic chunks
* 🧠 Generate embeddings using Sentence Transformers
* 🔍 Retrieve relevant chunks using FAISS
* 🤖 Generate context-aware answers using Cohere
* 💬 Interactive Streamlit interface
* 📝 Conversation history
* ⚡ Fast local vector search (No Pinecone required)

---

## Technology Stack

* Python
* Streamlit
* Cohere API
* FAISS
* Sentence Transformers
* LangChain Text Splitters
* PyMuPDF

---

## Project Workflow

```text
PDF Upload
     │
     ▼
Text Extraction
     │
     ▼
Text Chunking
     │
     ▼
Sentence Embeddings
     │
     ▼
FAISS Vector Store
     │
     ▼
Similarity Search
     │
     ▼
Retrieved Context
     │
     ▼
Cohere Language Model
     │
     ▼
Generated Answer
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/RAG_Document_Question_Answering.git
cd RAG_Document_Question_Answering
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**macOS/Linux**

```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Get a Cohere API Key

Create a free account from the Cohere Dashboard and generate your API key.

---

### 5. Run the Application

```bash
cd src
streamlit run app.py
```

Once the application starts, open your browser and navigate to:

**http://localhost:8501/**

---

## How to Use

1. Launch the Streamlit application.
2. Enter your Cohere API Key.
3. Upload a PDF document.
4. Click **Process Document**.
5. Ask questions related to the uploaded PDF.
6. View the generated answer and retrieved document chunks.

---

## Project Structure

```text
RAG_Document_Question_Answering/
│
├── src/
│   ├── app.py
│   ├── chatbot.py
│   └── vectorstore.py
│
├── requirements.txt
├── README.md
└── uploaded_document.pdf
```

---

## Future Improvements

* Support multiple PDF documents
* Display page numbers for retrieved chunks
* Add chat export (PDF/TXT)
* Add document summarization
* Support Word and text documents
* Deploy on Streamlit Community Cloud

---

## Learning Outcomes

This project demonstrates:

* Retrieval-Augmented Generation (RAG)
* Document preprocessing
* Text chunking
* Semantic embeddings
* Vector similarity search using FAISS
* Context-aware response generation
* Building AI applications with Streamlit

---

## License

This project is licensed under the Apache License.

---

## Acknowledgements

* Cohere for language models
* FAISS for efficient vector similarity search
* Sentence Transformers for semantic embeddings
* LangChain for text chunking
* Streamlit for the web interface
