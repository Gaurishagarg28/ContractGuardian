import pandas as pd
FILE_PATH = "data/datasets/cuad/master_clauses.csv"
def inspect_cuad():
    print("Loading CUAD master clauses...")
    df = pd.read_csv(FILE_PATH)
    print("\n================================")
    print("CUAD DATASET INFORMATION")
    print("================================")
    print("\nShape:")
    print(df.shape)
    print("\nColumns:")
    for i, column in enumerate(df.columns):
        print(i, "->", column)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())
if __name__ == "__main__":
    inspect_cuad()