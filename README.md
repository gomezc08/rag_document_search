---
title: conversation_rag_agent
app_file: app.py
sdk: gradio
sdk_version: 6.5.1
---

# RAG Document Search

Chat with an agent grounded in your documents. The app uses RAG (retrieval-augmented generation) with a ReAct agent: it retrieves relevant chunks from a vector store (FAISS) and injects them into the prompt so answers are based on your PDFs, URLs, or text files.

**[Try the app on Hugging Face](https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME)** *(replace with your Space URL after deploy)*

---

## Run locally

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (or use `pip` with the same deps from `pyproject.toml`)

### Setup

```bash
# Clone and enter the repo
cd rag_document_search

# Install dependencies with uv
uv sync

# Copy env and add your keys
cp .env.example .env
# Edit .env: OPENAI_API_KEY, NAME (e.g. "Chris Gomez")
```

### .env

Create a `.env` file (or copy from `.env.example` if you add one) with:

- `OPENAI_API_KEY` – required for embeddings and the chat model  
- `NAME` – name used in the chat persona (e.g. for “tell me about yourself”)

### Launch

```bash
uv run python app.py
```

Then open the Gradio URL (e.g. http://127.0.0.1:7860).

### Deploy to Hugging Face

```bash
uv run gradio deploy
```

Follow the prompts to create or update a Space. The frontmatter at the top of this README is used by Gradio/HF for the Space.

---

## Project layout

| Path | Purpose |
|------|--------|
| `app.py` | Entry point: load docs, build vector store, wire Chat to Gradio |
| `src/chat.py` | `Chat` class; bridges Gradio and RAG agent |
| `src/config/config.py` | Config and `DEFAULT_URLS` for ingestion |
| `src/document_ingestion/document_processor.py` | Load URLs/PDFs/txt, chunk with RecursiveCharacterTextSplitter |
| `src/vector_store/vector_store.py` | FAISS + OpenAI embeddings, retriever |
| `src/nodes/reactnode.py` | RAG + ReAct agent: retrieve → inject context → LLM |
| `src/state/rag_state.py` | RAG state (question, retrieved_docs, answer) |
| `data/` | Default location for PDFs; URLs set in `Config.DEFAULT_URLS` |

---

## How it works

1. **Ingestion** – `DocumentProcessor` loads from URLs and/or `data/` (PDFs, txt), chunks with overlap.
2. **Index** – Chunks are embedded (OpenAI) and stored in FAISS; a retriever is created.
3. **Chat** – Each user message is used to retrieve relevant chunks; those are formatted and prepended as “Relevant context from your documents” so the ReAct agent answers from your data. The agent also has a retriever tool for follow-up.

The name shown in the chat comes from `NAME` in `.env`, passed through from `app.py` into the agent.
