from risk_engine import calculate_risk


result = calculate_risk(
    "Renewal Term",
    0.3668
)


print("\n")
print("=" * 60)
print("CONTRACTGUARDIAN RISK RESULT")
print("=" * 60)

print(
    "Clause:",
    result["clause"]
)

print(
    "Confidence:",
    result["confidence"] * 100,
    "%"
)

print(
    "Base Risk:",
    result["base_score"]
)

print(
    "Risk Score:",
    result["risk_score"]
)

print(
    "Severity:",
    result["severity"]
)

print(
    "Reason:",
    result["reason"]
)

print("=" * 60)