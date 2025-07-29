import json
import torch
import requests
import numpy as np

from tqdm import tqdm
from pathlib import Path
from functools import reduce
from dataclasses import dataclass
from typing import Optional, Union
from clinical_trial_matcher.model.embeddings import BertEmbedding

class DataLoader:
    def __init__(self, data: list):
        self.data = data

    @classmethod
    def from_api(
            cls, api_url,
            fmt: Optional[str]='json',
            condition: Optional[str]='cancer',
            page_size: Optional[int]=1000,
            num_pages: Optional[int]=10
    ):
        params = {
            "format": fmt,
            "query.cond": condition,
            "pageSize": page_size,
        }
        studies = []
        # url = "https://clinicaltrials.gov/api/v2/studies"
        r = requests.get(api_url, params=params)
        data = r.json()
        studies.extend(data["studies"])
        for i in tqdm(range(num_pages), desc="Loading studies page by page"):
            params["pageToken"] = data["nextPageToken"]
            r = requests.get(api_url, params=params)
            data = r.json()
            studies.extend(data["studies"])
        return cls(data=studies)

    @classmethod
    def from_file(cls, file_path: Path):
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls(data=data)

    def __iter__(self):
        for item in self.data:
            yield item

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def save_to_file(self, file_path: Path):
        with open(file_path, "w") as f:
            json.dump(self.data, f)


@dataclass
class EmbeddingResult:
    data_description: str
    reference_embeddings: torch.Tensor
    similarity_scores: np.ndarray = None
    top_k_indices: np.ndarray = None
    top_k_scores: np.ndarray = None

    def compute_similarity(self, text_embedding: torch.Tensor) -> None:
        similarities = torch.nn.functional.cosine_similarity(
            text_embedding.unsqueeze(0),
            self.reference_embeddings,
            dim=1
        ).numpy()
        self.similarity_scores = similarities.flatten()

    def compute_top_k_similarities(self, text_embedding: torch.Tensor, k: int = 5) -> None:
        similarities = self.compute_similarity(text_embedding)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        self.top_k_indices, self.top_k_scores = top_k_indices, similarities[top_k_indices]


class EmbeddedData:
    def __init__(self, data_loader: DataLoader, embedding_model: Optional[BertEmbedding] = None):
        self.data_loader = data_loader
        if embedding_model is None:
            self.embedding_model = BertEmbedding()
        else:
            self.embedding_model = BertEmbedding(embedding_model)

    def embed_for_key(self, key: Union[str, list[str]]):
        if isinstance(key, str):
            key = [key]
        embeddings = []
        for item in tqdm(self.data_loader, desc=f"Embedding data from key {[k for k in key]}"):
            text = reduce(dict.__getitem__, key, item)
            embeddings.append(self.embedding_model.embed(text))
        return torch.stack(embeddings)
