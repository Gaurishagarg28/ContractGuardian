import pandas as pd
FILE_PATH = "data/datasets/cuad/master_clauses.csv"
df = pd.read_csv(FILE_PATH)
print("=" * 80)
print("CUAD LABEL INSPECTION")
print("=" * 80)
categories = [
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Governing Law",
    "Non-Compete",
    "Exclusivity",
    "Termination For Convenience",
    "Ip Ownership Assignment",
    "Uncapped Liability",
    "Cap On Liability",
    "Insurance"
]
for category in categories:
    possible_columns = [
        col for col in df.columns
        if col.strip().lower() == f"{category}-answer".lower()
    ]

    if not possible_columns:
        print(f"\n[WARNING] Answer column not found: {category}")
        continue
    
    answer_col = f"{category}-Answer"
    print("\n" + "=" * 80)
    print(f"CATEGORY: {category}")
    print("=" * 80)
    print("\nAnswer distribution:")
    print(df[answer_col].value_counts(dropna=False).head(10))
    print("\nExamples:")

    valid = df[
        df[category].notna() &
        (df[category].astype(str).str.strip() != "")
    ]
    for _, row in valid.head(3).iterrows():
        print("\n--- Contract:", row["Filename"])
        print("Context:")
        print(str(row[category])[:1000])
        print("\nAnswer:")
        print(row[answer_col])