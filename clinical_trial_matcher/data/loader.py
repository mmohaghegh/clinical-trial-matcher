import json
import pickle

import torch
import requests
import numpy as np

from tqdm import tqdm
from pathlib import Path
from functools import reduce
from dataclasses import dataclass
from typing import Optional, Union, Generator

from clinical_trial_matcher.model.embeddings import BertEmbedding

class DataLoader:
    def __init__(
            self,
            data: list,
            page_size: Optional[int] = None,
            page_num: Optional[int] = None,
            api_url: Optional[str] = None,
            condition: Optional[str] = None,
            next_page_token: Optional[str] = None
    ):
        self.data = data
        self.page_size = page_size
        self.page_num = page_num
        self.api_url = api_url
        self.condition = condition
        self.next_page_token = next_page_token
        self.ids: list[str] = list()
        self._extract_ids()

    def _extract_ids(self):
        for item in self.data:
            self.ids.append(item['protocolSection']['identificationModule']['nctId'])

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
        return cls(
            data=studies,
            page_size=page_size,
            page_num=num_pages,
            api_url=api_url,
            condition=condition,
            next_page_token=data.get("nextPageToken", None)
        )

    @classmethod
    def from_file(cls, file_path: Path):
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls(data=data)

    def get_ntcIds_from_indices(self, indices: list[int]) -> list[str]:
        """
        Get NCT IDs from the data based on provided indices.

        Args:
            indices (list[int]): List of indices to retrieve NCT IDs from.

        Returns:
            list[str]: List of NCT IDs corresponding to the provided indices.
        """
        return [self.ids[i] for i in indices if i < len(self.ids)]

    def generate_entire_data_from_nctIds(self, nct_ids: list[str], fmt: str = 'json') -> Generator[dict, None, None]:
        if self.api_url is None:
            raise ValueError("DataLoader instance is not generated from API, hence cannot fetch data by NCT IDs.")
        for nct_id in nct_ids:
            params = {
                "format": fmt,
                "query.id": nct_id,
            }
            r = requests.get(self.api_url, params=params)
            if r.status_code != 200:
                raise ValueError(f"Failed to fetch data for NCT ID {nct_id}. Status code: {r.status_code}")
            data = r.json()
            yield data

    def generate_entire_data_from_indices(self, indices: list[int], fmt: str = 'json') -> Generator[dict, None, None]:
        """
        Generate entire data from the indices of the DataLoader instance.

        Args:
            fmt (str): Format of the data to be fetched. Default is 'json'.

        Yields:
            dict: Data corresponding to each index in the DataLoader.
        """
        ntcIds = self.get_ntcIds_from_indices(indices)
        if not ntcIds:
            raise ValueError("No NCT IDs found for the provided indices.")
        return self.generate_entire_data_from_nctIds(ntcIds, fmt=fmt)

    def save_api_state(self, path: Path):
        if self.api_url is None:
            raise ValueError("DataLoader instance is not generated from API, hence cannot save API state.")
        data_loader = DataLoader(
            data=list(),
            page_size=self.page_size,
            page_num=self.page_num,
            api_url=self.api_url,
            condition=self.condition,
            next_page_token=self.next_page_token
        )
        data_loader.save_to_file(path)

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
    embedding_model: BertEmbedding = None
    model_name: str = None

    def tokenize_and_embed(self, text: str) -> torch.Tensor:
        if self.embedding_model is None:
            self.embedding_model = BertEmbedding(self.model_name)
        return self.embedding_model.embed(text)

    def compute_similarity(self, text: str) -> np.ndarray:
        text_embedding = self.tokenize_and_embed(text)
        similarities = torch.nn.functional.cosine_similarity(
            text_embedding.unsqueeze(0),
            self.reference_embeddings,
            dim=1
        ).numpy()
        return similarities.flatten()

    def compute_top_k_similarities(self, text: str, k: int = 5) -> None:
        similarities = self.compute_similarity(text)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        self.top_k_indices, self.top_k_scores = top_k_indices, similarities[top_k_indices]

@dataclass
class EmbeddingResults:
    data_description: list[str]
    embedding_results: list[EmbeddingResult]

    @classmethod
    def from_a_embedding_result(cls, embedding_result: EmbeddingResult):
        return cls(
            data_description=[embedding_result.data_description],
            embedding_results=[embedding_result]
        )

    # Method to add a new embedding result
    def add_embedding_result(self, data_description: str, reference_embeddings: torch.Tensor, model_name: str) -> None:
        result = EmbeddingResult(
            data_description=data_description,
            reference_embeddings=reference_embeddings,
            model_name=model_name
        )
        self.embedding_results.append(result)
        self.data_description.append(data_description)

    def get_similarity_scores(self, text: str) -> np.ndarray:
        similarities = []
        for result in self.embedding_results:
            similarities.append(result.compute_similarity(text))
        return np.mean(similarities, axis=0)

    def get_top_k_indices_based_on_similarity(self, text: str, k: int = 5) -> np.ndarray:
        avg_similarities = self.get_similarity_scores(text)
        top_k_indices = np.argsort(avg_similarities)[-k:][::-1]
        return top_k_indices



    def write_to_file(self, file_path: Path):
        with open(file_path, "wb") as f:
            pickle.dump(self.embedding_results, f)

class EmbeddedData:
    def __init__(self, data_loader: DataLoader, model_name = None):
        self.data_loader = data_loader
        if model_name is None:
            self.embedding_model = BertEmbedding()
        else:
            self.embedding_model = BertEmbedding(model_name)

    def embed_for_key(self, key: Union[str, list[str]]):
        if isinstance(key, str):
            key = [key]
        embeddings = []
        for item in tqdm(self.data_loader, desc=f"Embedding data from key {[k for k in key]}"):
            text = reduce(dict.__getitem__, key, item)
            embeddings.append(self.embedding_model.embed(text))
        return torch.stack(embeddings)
