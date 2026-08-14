import os
import io
from PIL import Image

try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False


def extract_text_from_image(image_bytes_or_pil):
    """
    Perform OCR on an image (PIL Image or raw bytes).
    Fallback gracefully if OCR binaries are missing.
    """
    if isinstance(image_bytes_or_pil, bytes):
        image = Image.open(io.BytesIO(image_bytes_or_pil))
    else:
        image = image_bytes_or_pil

    if HAS_PYTESSERACT:
        try:
            return pytesseract.image_to_string(image)
        except Exception as e:
            print(f"[OCR Warning] pytesseract failed: {e}")

    # Fallback message if OCR engine unavailable
    return ""


def process_scanned_page(pymupdf_page):
    """
    Renders PyMuPDF page to pixmap image and runs OCR if text extraction was empty.
    """
    pix = pymupdf_page.get_pixmap(dpi=150)
    img_data = pix.tobytes("png")
    text = extract_text_from_image(img_data)
    return text
