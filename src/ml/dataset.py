import pandas as pd
import torch

from torch.utils.data import Dataset
from transformers import AutoTokenizer


MODEL_NAME = "bert-base-uncased"


class ContractClauseDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        label_encoder,
        max_length=256
    ):

        self.data = dataframe.reset_index(
            drop=True
        )

        self.tokenizer = tokenizer

        self.label_encoder = label_encoder

        self.max_length = max_length


    def __len__(self):

        return len(self.data)


    def __getitem__(self, index):

        row = self.data.iloc[index]

        text = str(
            row["clause_text"]
        )

        label = self.label_encoder.transform(
            [row["clause_type"]]
        )[0]

        encoding = self.tokenizer(
            text,

            truncation=True,

            padding="max_length",

            max_length=self.max_length,

            return_tensors="pt"
        )

        return {

            "input_ids":
                encoding["input_ids"].squeeze(0),

            "attention_mask":
                encoding["attention_mask"].squeeze(0),

            "labels":
                torch.tensor(
                    label,
                    dtype=torch.long
                )
        }