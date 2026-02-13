from pathlib import Path

import fitz  # pymupdf
import pytesseract
from PIL import Image
from langchain_core.documents import Document
from tqdm import tqdm

from src.config import OCR_CACHE_PATH


def extract_text_from_pdf(pdf_path: str) -> str:
    """Convert a scanned PDF to text using pymupdf + pytesseract OCR."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in tqdm(doc, desc="  Pages", unit="page", leave=False):
        # Render page at 300 DPI
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img)
        pages.append(text)
    doc.close()
    return "\n\n".join(pages)


def _cache_path_for(pdf_name: str) -> Path:
    return Path(OCR_CACHE_PATH) / f"{pdf_name}.txt"


def load_pdfs_as_documents(folder: str) -> list[Document]:
    """Load all PDFs in a folder, OCR them, and return LangChain Documents."""
    folder_path = Path(folder)
    pdf_files = sorted(folder_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {folder}")
        return []

    cache_dir = Path(OCR_CACHE_PATH)
    cache_dir.mkdir(parents=True, exist_ok=True)

    documents = []
    for pdf_file in tqdm(pdf_files, desc="Books", unit="book"):
        cache_file = _cache_path_for(pdf_file.name)

        if cache_file.exists():
            tqdm.write(f"  Cached: {pdf_file.name}")
            text = cache_file.read_text(encoding="utf-8")
        else:
            tqdm.write(f"  Processing: {pdf_file.name}")
            text = extract_text_from_pdf(str(pdf_file))
            cache_file.write_text(text, encoding="utf-8")
            tqdm.write(f"  Done — {len(text)} characters extracted")

        doc = Document(
            page_content=text,
            metadata={"source": pdf_file.name},
        )
        documents.append(doc)

    print(f"\nTotal: {len(documents)} document(s) loaded")
    return documents
