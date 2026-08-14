import os
import pymupdf

from pdf_loader import extract_text_from_pdf
from ocr import process_scanned_page


def parse_document(pdf_path, min_chars_per_page=30):
    """
    Parse a contract PDF, extracting text per page, running OCR fallback if necessary.

    Returns:
        dict containing:
            filename: str
            total_pages: int
            pages: list of dicts {page_number, text, is_ocr, char_count}
            full_text: str
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    filename = os.path.basename(pdf_path)
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)

    pages = []
    full_text_chunks = []

    for page_idx in range(total_pages):
        page = doc[page_idx]
        text = page.get_text("text") or ""
        is_ocr = False

        if len(text.strip()) < min_chars_per_page:
            ocr_text = process_scanned_page(page)
            if len(ocr_text.strip()) > len(text.strip()):
                text = ocr_text
                is_ocr = True

        page_data = {
            "page_number": page_idx + 1,
            "text": text,
            "is_ocr": is_ocr,
            "char_count": len(text)
        }
        pages.append(page_data)
        full_text_chunks.append(text)

    doc.close()

    return {
        "filename": filename,
        "total_pages": total_pages,
        "pages": pages,
        "full_text": "\n\n".join(full_text_chunks)
    }
