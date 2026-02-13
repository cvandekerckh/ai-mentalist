from pathlib import Path

import fitz  # pymupdf
import pytesseract
from PIL import Image
from langchain_core.documents import Document


def extract_text_from_pdf(pdf_path: str) -> str:
    """Convert a scanned PDF to text using pymupdf + pytesseract OCR."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, start=1):
        # Render page at 300 DPI
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img)
        pages.append(text)
        print(f"  OCR page {i}/{len(doc)}")
    doc.close()
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
