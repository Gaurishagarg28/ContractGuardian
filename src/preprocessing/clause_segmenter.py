import re
try:
    from src.preprocessing.text_cleaner import clean_text
except ImportError:
    from text_cleaner import clean_text



def segment_clauses(full_text_or_pages, min_clause_words=8):
    """
    Segments contract text into sections and clauses with page numbers and text snippets.

    Args:
        full_text_or_pages: either a string or list of dicts {page_number, text}

    Returns:
        list of dicts containing:
            clause_id: int (1-indexed)
            section_title: str
            page_number: int
            clause_text: str
            word_count: int
    """
    pages_list = []

    if isinstance(full_text_or_pages, str):
        pages_list.append({"page_number": 1, "text": full_text_or_pages})
    elif isinstance(full_text_or_pages, list):
        pages_list = full_text_or_pages
    else:
        raise ValueError("Invalid input format for segment_clauses")

    raw_clauses = []
    current_section = "General"

    # Regex patterns for section titles and clause boundaries
    section_pattern = re.compile(
        r"(?i)^(?:ARTICLE|SECTION|\d{1,2}\.)\s*([0-9A-Z\.\s\-–:]{3,60})$"
    )

    heading_split_pattern = re.compile(
        r"(?m)(?=(?:^(?:ARTICLE|SECTION)\s+\d+|^\d{1,2}\.\s+[A-Z]|^\(\s*[a-z0-9]+\s*\)\s+[A-Z]))"
    )

    clause_counter = 1

    for page_info in pages_list:
        p_num = page_info.get("page_number", 1)
        p_text = clean_text(page_info.get("text", ""))

        if not p_text:
            continue

        # Split by double newline or heading boundaries
        paragraphs = re.split(r"\n\s*\n", p_text)

        for para in paragraphs:
            para_str = para.strip()
            if not para_str:
                continue

            # Check if this paragraph is a standalone section title
            sec_match = section_pattern.match(para_str)
            if sec_match and len(para_str.split()) < 10:
                current_section = para_str
                continue

            # Check if paragraph contains multiple numbered clauses
            sub_chunks = heading_split_pattern.split(para_str)

            for chunk in sub_chunks:
                chunk_cleaned = chunk.strip()
                words = chunk_cleaned.split()

                if len(words) >= min_clause_words:
                    raw_clauses.append({
                        "clause_id": clause_counter,
                        "section_title": current_section,
                        "page_number": p_num,
                        "clause_text": chunk_cleaned,
                        "word_count": len(words)
                    })
                    clause_counter += 1

    # Fallback if no clauses segmented (e.g. single unformatted block)
    if not raw_clauses and pages_list:
        combined = " ".join([p.get("text", "") for p in pages_list])
        cleaned = clean_text(combined)
        words = cleaned.split()

        # Split into ~100 word chunks
        chunk_size = 100
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) >= min_clause_words:
                raw_clauses.append({
                    "clause_id": clause_counter,
                    "section_title": "General",
                    "page_number": 1,
                    "clause_text": " ".join(chunk_words),
                    "word_count": len(chunk_words)
                })
                clause_counter += 1

    return raw_clauses
