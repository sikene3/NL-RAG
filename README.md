# NL RAG — Local Retrieval-Augmented Generation System

**A fully local RAG system — no cloud, no API keys, no data leakage.**

---

## Overview

A fully offline RAG system running entirely on your machine. Supports:
- **PDF ingestion** → text chunking → vector storage in Chroma DB
- **RAG chat** → semantic retrieval → answer generation via Ollama
- **Data analysis (CSV/Excel)** → Pandas Agent for analytical queries
- **Interactive web UI** → ChatGPT-like Streamlit interface

---

## Project Structure

```
NL RAG/
├── src/                          # Source code
│   ├── ingest_pdf.py             # Phase 1: PDF ingestion & vector storage
│   ├── chat_pdf.py               # Phase 2: Terminal-based RAG chat
│   └── app_rag.py                # Phase 3-5: Full Streamlit web app
├── data/                         # Data directory
│   ├── document.pdf              # Default PDF (place your files here)
│   └── chroma_db/                # Chroma vector store (auto-created)
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

---

## Prerequisites

| Component | Purpose |
|-----------|---------|
| **Python 3.10+** | Core runtime |
| **Ollama** | Local LLM inference server |
| **llama3.3:latest** (or any model) | LLM for retrieval & analysis |

### Install Ollama

1. Download from [ollama.com](https://ollama.com)
2. Start the server: `ollama serve`
3. Pull the model: `ollama pull llama3.3:latest`

---

## Installation

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate it
# Linux / macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Phase 1: Ingest PDF

Convert any PDF into searchable vectors:

```bash
cd src
python ingest_pdf.py
```

**Note:** Place your PDF at `data/document.pdf` or edit the path in `ingest_pdf.py`.

---

### Phase 2: Terminal Chat

Interactive terminal chat to query your document:

```bash
cd src
python chat_pdf.py
```

- Type your question and press Enter
- Type `exit` or `quit` to leave
- Source chunks displayed with each answer

---

### Phase 3-5: Full Web App

Full web interface supporting:
- **Document Q&A (RAG):** Upload PDFs and ask questions
- **Data Analysis (Agent):** Upload CSV/Excel and analyze data

```bash
cd src
streamlit run app_rag.py
```

---

## How It Works

```
┌─────────────┐     ┌──────────────────┐     ┌────────────┐
│  PDF/CSV/   │────▶│  Text Splitter /  │────▶│  Chroma DB │
│  Excel File │     │  Pandas Loader    │     │  (Vector   │
└─────────────┘     └──────────────────┘     │  Store)    │
                                             └─────┬──────┘
                                                   │
┌─────────────┐     ┌──────────────────┐           │
│  User       │────▶│  Semantic        │◀──────────┘
│  Question   │     │  Retrieval (k=3) │
└─────────────┘     └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Context + Prompt │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Ollama LLM      │
                    │  (llama3.3)      │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Generated       │
                    │  Answer          │
                    └──────────────────┘
```

---

## Configuration

You can modify these constants in each script:

| Constant | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `OLLAMA_MODEL` | `llama3.3:latest` | LLM model |
| `CHUNK_SIZE` | `1000` | Chunk size (chars) |
| `CHUNK_OVERLAP` | `200` | Chunk overlap |
| `RETRIEVAL_K` | `3` | Retrieved chunks per query |

---

## FAQ

**Q: I get "ModuleNotFoundError: No module named 'langchain_experimental'"?**
A: Install it: `pip install langchain-experimental openpyxl`

**Q: Error "Expected Embeddings to be non-empty list"?**
A: The PDF may be scanned/image-based. Use a PDF with extractable text.

**Q: Does it work offline?**
A: Yes, everything is local. Internet needed only once to download the model.

**Q: Can I use a different LLM model?**
A: Yes, pull any Ollama model and change `OLLAMA_MODEL` in the scripts.

---

## License

MIT — Free to use for any purpose.

---

**Built with**  
LangChain • Chroma • HuggingFace • Ollama • Streamlit • PyTorch
