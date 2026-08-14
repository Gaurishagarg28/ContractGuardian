import torch
import torch.nn as nn

from transformers import AutoModel


class ContractBERT(nn.Module):

    def __init__(self, model_name):

        super().__init__()

        self.bert = AutoModel.from_pretrained(
            model_name
        )

        # BERT is only a feature extractor
        for param in self.bert.parameters():
            param.requires_grad = False

        self.bert.eval()


    @torch.no_grad()
    def get_embeddings(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS embedding
        embeddings = (
            outputs.last_hidden_state[:, 0, :]
        )

        return embeddings