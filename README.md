# AI Mentalist

A virtual mentalist chatbot that uses RAG (Retrieval-Augmented Generation) over scanned mentalism PDF books. Chat with a mentalist persona in a Streamlit web app, with responses grounded in your book collection.

## Stack

OpenAI API + LangChain + FAISS + pytesseract (OCR) + Streamlit

## Prerequisites

Install system dependencies (macOS):

```bash
brew install tesseract
```

Install Python dependencies:

```bash
uv sync
```

## Setup

1. Copy the environment template and add your OpenAI API key:

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

2. Place your scanned mentalism PDF books in the `pdfs/` folder:

```bash
mkdir -p pdfs
# Copy your PDF files into pdfs/
```

3. Run the ingestion pipeline (OCR + indexing):

```bash
uv run python ingest.py
```

This will OCR all PDFs and create a FAISS vector index in `faiss_index/`.

4. Launch the chatbot:

```bash
uv run streamlit run app.py
```

## Configuration

All settings can be customized in `.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Your OpenAI API key (required) |
| `PDF_FOLDER_PATH` | `./pdfs` | Folder containing PDF books |
| `FAISS_INDEX_PATH` | `./faiss_index` | Where to save/load the vector index |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `CHUNK_SIZE` | `1000` | Text chunk size for splitting |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
