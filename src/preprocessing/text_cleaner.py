import re


def clean_text(raw_text):
    """
    Clean extracted contract text by stripping headers, page numbers,
    normalizing whitespace, while preserving legal structure and punctuation.
    """
    if not raw_text:
        return ""

    text = raw_text

    # Remove common page number patterns like "Page 1 of 10", "- Page 2 -", "Page 3"
    text = re.sub(r"(?i)\bpage[ \t]+\d+[ \t]+of[ \t]+\d+\b", "", text)
    text = re.sub(r"(?i)\bpage[ \t]+\d+\b", "", text)
    text = re.sub(r"(?m)^\s*[-—]\s*\d+\s*[-—]\s*$", "", text)

    # Remove repetitive header/footer tags like "CONFIDENTIAL", "DRAFT"
    text = re.sub(r"(?i)\bconfidential\b", "", text)
    text = re.sub(r"(?i)\bexecution copy\b", "", text)

    # Normalize excessive carriage returns and non-breaking spaces
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ")

    # Remove multi-blank lines (more than 2 newlines in a row)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Replace multiple spaces with a single space (while keeping single newlines)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned_text = "\n".join(lines)

    return cleaned_text.strip()
