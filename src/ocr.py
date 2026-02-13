import os
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path
from langchain_core.documents import Document


def extract_text_from_pdf(pdf_path: str) -> str:
    """Convert a scanned PDF to text using pdf2image + pytesseract OCR."""
    images = convert_from_path(pdf_path, dpi=300)
    pages = []
    for i, image in enumerate(images, start=1):
        text = pytesseract.image_to_string(image)
        pages.append(text)
        print(f"  OCR page {i}/{len(images)}")
    return "\n\n".join(pages)


def load_pdfs_as_documents(folder: str) -> list[Document]:
    """Load all PDFs in a folder, OCR them, and return LangChain Documents."""
    folder_path = Path(folder)
    pdf_files = sorted(folder_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {folder}")
        return []

    documents = []
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        text = extract_text_from_pdf(str(pdf_file))
        doc = Document(
            page_content=text,
            metadata={"source": pdf_file.name},
        )
        documents.append(doc)
        print(f"  Done — {len(text)} characters extracted")

    print(f"\nTotal: {len(documents)} document(s) loaded")
    return documents
