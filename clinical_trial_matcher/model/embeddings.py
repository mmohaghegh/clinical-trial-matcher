import torch
import numpy as np
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel


class BertEmbedding:
    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use CLS token representation
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        return cls_embedding

    def compute_reference_embedding(self, reference_texts: list[str]) -> None:
        embeddings = [self.embed(text) for text in reference_texts]
        self.reference_embedding = torch.stack(embeddings)

    def set_reference_embedding(self, reference_embeddings: list[torch.Tensor]) -> None:
        self.reference_embedding = torch.stack(reference_embeddings)