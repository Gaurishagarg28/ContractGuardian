from transformers import AutoTokenizer


MODEL_NAME = "bert-base-uncased"


print("Loading BERT tokenizer...")


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


text = """
This agreement shall automatically renew
for successive one year periods unless either
party provides written notice of termination.
"""


encoding = tokenizer(

    text,

    truncation=True,

    padding="max_length",

    max_length=256,

    return_tensors="pt"
)


print("\nTokenizer loaded successfully.")


print("\nInput IDs shape:")

print(
    encoding["input_ids"].shape
)


print("\nAttention mask shape:")

print(
    encoding["attention_mask"].shape
)


print("\nFirst 50 tokens:")

print(
    tokenizer.convert_ids_to_tokens(
        encoding["input_ids"][0]
    )[:50]
)