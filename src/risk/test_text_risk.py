from text_risk_analyzer import analyze_text


clause = """
The license granted under this Agreement shall be
perpetual, irrevocable and worldwide. The Licensee
may use the Software without limitation and without
obtaining further consent from the Licensor.
"""


result = analyze_text(
    clause
)


print("\n")
print("=" * 65)
print("TEXT RISK ANALYSIS")
print("=" * 65)

print(
    "\nRisk Points:",
    result["risk_points"]
)

print(
    "\nMatched Text:"
)

for item in result["matched_text"]:
    print(
        " -",
        item
    )

print(
    "\nRisk Indicators:"
)

for indicator in result["indicators"]:

    print(
        f'\n{indicator["indicator"]}'
    )

    print(
        f'Points: {indicator["points"]}'
    )

    print(
        f'Description: '
        f'{indicator["description"]}'
    )

print("=" * 65)