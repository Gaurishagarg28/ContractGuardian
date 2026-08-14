import re


# ============================================================
# RISK INDICATORS
# ============================================================

RISK_INDICATORS = {

    "unlimited": {
        "patterns": [
            r"\bunlimited\b",
            r"\bwithout limitation\b",
            r"\bwithout any limit\b"
        ],
        "points": 20,
        "description": "Potentially unlimited obligation or exposure."
    },

    "perpetual": {
        "patterns": [
            r"\bperpetual\b",
            r"\bforever\b",
            r"\bindefinitely\b"
        ],
        "points": 20,
        "description": "Obligation or right may continue indefinitely."
    },

    "irrevocable": {
        "patterns": [
            r"\birrevocable\b",
            r"\birrevocably\b"
        ],
        "points": 15,
        "description": "The right may be difficult or impossible to revoke."
    },

    "sole_discretion": {
        "patterns": [
            r"\bsole discretion\b",
            r"\babsolute discretion\b",
            r"\bsole and absolute discretion\b"
        ],
        "points": 15,
        "description": "One party may have broad unilateral discretion."
    },

    "automatic_renewal": {
        "patterns": [
            r"\bautomatically renew\b",
            r"\bautomatically renewed\b",
            r"\bautomatic renewal\b",
            r"\brenew automatically\b"
        ],
        "points": 10,
        "description": "The agreement may renew without a new agreement."
    },

    "worldwide": {
        "patterns": [
            r"\bworldwide\b",
            r"\bworld[- ]wide\b",
            r"\bthroughout the world\b"
        ],
        "points": 10,
        "description": "The provision may apply across a broad geographic scope."
    },

    "without_consent": {
        "patterns": [
            r"\bwithout (?:prior )?consent\b",
            r"\bwithout the consent\b",
            r"\bwithout obtaining consent\b"
        ],
        "points": 10,
        "description": "A party may act without obtaining the other party's consent."
    },

    "at_any_time": {
        "patterns": [
            r"\bat any time\b",
            r"\bat any time and from time to time\b"
        ],
        "points": 8,
        "description": "The provision may permit action at a broad or undefined time."
    },

    "immediately": {
        "patterns": [
            r"\bimmediately\b",
            r"\bwith immediate effect\b"
        ],
        "points": 5,
        "description": "The provision may impose an immediate obligation or consequence."
    },

    "sole_remedy": {
        "patterns": [
            r"\bsole remedy\b",
            r"\bexclusive remedy\b"
        ],
        "points": 12,
        "description": "Available remedies may be restricted."
    },

    "waiver": {
        "patterns": [
            r"\bwaive\b",
            r"\bwaiver\b",
            r"\bwaived\b"
        ],
        "points": 8,
        "description": "A party may be giving up a contractual or legal right."
    },

    "indemnify": {
        "patterns": [
            r"\bindemnify\b",
            r"\bindemnification\b",
            r"\bhold harmless\b"
        ],
        "points": 12,
        "description": "The provision may create indemnification obligations."
    },

    "liquidated_damages": {
        "patterns": [
            r"\bliquidated damages\b"
        ],
        "points": 15,
        "description": "A predefined financial consequence may apply."
    },

    "termination_without_cause": {
        "patterns": [
            r"\bterminate.*without cause\b",
            r"\btermination.*without cause\b",
            r"\bterminate.*for convenience\b"
        ],
        "points": 10,
        "description": "A party may have broad termination rights."
    }
}


# ============================================================
# ANALYZER
# ============================================================

def analyze_text(clause_text):

    if not clause_text:
        return {
            "risk_points": 0,
            "indicators": [],
            "matched_text": []
        }


    text = clause_text.lower()

    indicators = []
    matched_text = []

    total_points = 0


    for name, rule in RISK_INDICATORS.items():

        for pattern in rule["patterns"]:

            matches = re.findall(
                pattern,
                text,
                flags=re.IGNORECASE
            )

            if matches:

                indicators.append({

                    "indicator": name,

                    "points": rule["points"],

                    "description":
                        rule["description"]
                })


                matched_text.extend(
                    matches
                )

                total_points += (
                    rule["points"]
                )

                # Don't count the same
                # indicator more than once.
                break


    # Cap the textual contribution
    total_points = min(
        total_points,
        60
    )


    return {

        "risk_points":
            total_points,

        "indicators":
            indicators,

        "matched_text":
            list(set(matched_text))
    }