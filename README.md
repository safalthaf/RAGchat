# RAGchat
A Retrieval-Augmented Generation (RAG) chatbot built with **LangGraph**, using local **Hugging Face** embedding models for dense vector generation and Google's **Gemini API** for response generation.

## Features

* **Local Vector Embeddings**: Uses `BAAI/bge-small-en-v1.5` via Hugging Face to generate 384-dimensional dense vectors locally—no API rate limits or costs during document ingestion.
* **ChromaDB Integration**: Persistent, high-performance local vector database storage.
* **LangGraph Pipeline**: Modular graph-based execution state machine controlling document retrieval and response generation nodes.
* **Gemini LLM**: Powered by `gemini-3.5-flash-lite` for accurate, fast context-aware query responses.
* **Multi-Format Ingestion**: Supports `.pdf`, `.docx`, and `.txt` files with recursive character splitting.

## Architecture Flow

```text
[ Document Folder ] ──► [ Local Embedding (BGE-Small) ] ──► [ Chroma Vector DB ]
                                                                   │
[ User Input ] ──► [ LangGraph Node: Retrieve ] ◄──────────────────┘
                         │
                         ▼
                   [ LangGraph Node: Generate ] ──► [ Gemini API ] ──► [ Output ]

