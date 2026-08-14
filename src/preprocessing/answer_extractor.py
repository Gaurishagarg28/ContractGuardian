import re


def extract_key_values(clause_text):
    """
    Extract key legal entities and values from a clause text:
    - dates
    - durations / notice periods
    - monetary amounts / caps
    - parties mentioned
    - rights and obligations

    Returns:
        dict containing extracted lists for each category
    """
    if not clause_text:
        return {
            "dates": [],
            "durations": [],
            "amounts": [],
            "parties": [],
            "has_obligations": False,
            "has_rights": False
        }

    text = clause_text

    # 1. Dates
    date_patterns = [
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b",
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"
    ]
    dates = []
    for pattern in date_patterns:
        found = re.findall(pattern, text, flags=re.IGNORECASE)
        dates.extend(found)

    # 2. Durations / Notice Periods
    duration_pattern = r"\b\d+\s+(?:business\s+)?(?:day|days|week|weeks|month|months|year|years|hour|hours)\b"
    durations = re.findall(duration_pattern, text, flags=re.IGNORECASE)

    # 3. Monetary Amounts & Caps
    amount_patterns = [
        r"\$\s*[\d,]+(?:\.\d{2})?\b",
        r"\b(?:USD|EUR|GBP|INR|Rs\.?)\s*[\d,]+(?:\.\d{2})?\b",
        r"\b\d+%\s+of\s+[^.,;\n]+\b",
        r"\b\d+\s+percent\b"
    ]
    amounts = []
    for pattern in amount_patterns:
        found = re.findall(pattern, text, flags=re.IGNORECASE)
        amounts.extend(found)

    # 4. Key Parties
    party_pattern = r"\b(?:Licensor|Licensee|Disclosing Party|Receiving Party|Buyer|Seller|Company|Client|Contractor|Provider|Customer)\b"
    parties = list(set(re.findall(party_pattern, text, flags=re.IGNORECASE)))

    # 5. Obligations and Rights
    has_obligations = bool(re.search(r"\b(?:shall|must|agrees to|obligated|undertakes)\b", text, flags=re.IGNORECASE))
    has_rights = bool(re.search(r"\b(?:entitled to|shall have the right|may|reserves the right)\b", text, flags=re.IGNORECASE))

    return {
        "dates": list(set(dates)),
        "durations": list(set(durations)),
        "amounts": list(set(amounts)),
        "parties": parties,
        "has_obligations": has_obligations,
        "has_rights": has_rights
    }
