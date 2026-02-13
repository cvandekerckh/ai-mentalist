"""CLI script to run the OCR + indexing pipeline."""

from src.config import PDF_FOLDER_PATH
from src.indexer import create_vector_store
from src.ocr import load_pdfs_as_documents


def main():
    print("=== AI Mentalist — Ingestion Pipeline ===\n")

    print("Step 1: OCR — extracting text from PDFs...")
    documents = load_pdfs_as_documents(PDF_FOLDER_PATH)
    if not documents:
        print("No documents to index. Place PDF files in the pdfs/ folder.")
        return

    print("\nStep 2: Indexing — chunking + embedding + FAISS...")
    create_vector_store(documents)

    print("\nDone! You can now run the chatbot with: uv run streamlit run app.py")


if __name__ == "__main__":
    main()
