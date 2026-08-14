from datasets import load_dataset
def load_cuad():
    print("Loading CUAD dataset...")
    dataset = load_dataset("theatticusproject/cuad")
    return dataset
def inspect_dataset(dataset):
    print("\n========== DATASET ==========")
    print(dataset)
    for split in dataset:
        print(f"\n========== {split.upper()} ==========")
        print("Number of examples:", len(dataset[split]))
        print("Columns:", dataset[split].column_names)
        print("\nFirst example:")
        print(dataset[split][0])
if __name__ == "__main__":
    dataset = load_cuad()
    inspect_dataset(dataset)