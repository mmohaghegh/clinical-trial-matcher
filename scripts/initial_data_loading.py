import json
import requests
from pathlib import Path

from tqdm import tqdm

from clinical_trial_matcher.data.loader import EmbeddedData, DataLoader, EmbeddingResult, EmbeddingResults

READ_EXISTING_DATA = False
STORE_DATA = True
MODEL_NAME = 'dmis-lab/biobert-base-cased-v1.1'
data_path = Path('./data')
file_path = data_path / "clinical_trials.json"

if READ_EXISTING_DATA:
    data_obj = DataLoader.from_file(file_path)
    print(f"Loaded {len(data_obj)} studies from existing data.")
else:
    params = {
        "format": "json",
        "query.cond": "cancer",
        "pageSize": 10,
    }
    studies = []
    url = "https://clinicaltrials.gov/api/v2/studies"
    data_obj = DataLoader.from_api(
        api_url=url,
        fmt=params["format"],
        condition=params["query.cond"],
        page_size=params["pageSize"],
        num_pages=1
    )
    if STORE_DATA:
        data_obj.save_to_file(file_path)

embedding_data = EmbeddedData(data_obj, model_name=MODEL_NAME)
embedding_result = embedding_data.embed_for_key(['protocolSection', 'identificationModule', 'briefTitle'])
embedding_results = EmbeddingResults.from_a_embedding_result(
    EmbeddingResult(data_description='brief title of clinical trials', reference_embeddings=embedding_result, model_name=MODEL_NAME),
)
embedding_results.add_embedding_result(
    data_description='brief summary of clinical trials',
    reference_embeddings=embedding_data.embed_for_key(['protocolSection', 'descriptionModule', 'briefSummary']),
    model_name=MODEL_NAME
)
embedding_results.add_embedding_result(
    data_description='eligibility criteria of clinical trials',
    reference_embeddings=embedding_data.embed_for_key(['protocolSection', 'eligibilityModule', 'eligibilityCriteria']),
    model_name=MODEL_NAME
)
# embedding_results.add_embedding_result(
#     data_description='conditions of clinical trials',
#     reference_embeddings=embedding_data.embed_for_key(['protocolSection', 'conditionsModule', 'conditions'])
# )
query_text = "breast cancer treatment"
top_5_indices = embedding_results.get_top_k_indices_based_on_similarity(query_text, k=5)
top_ntcIds = data_obj.get_ntcIds_from_indices(top_5_indices)
string_decorators = '#' * 50
for data in data_obj.generate_entire_data_from_nctIds(top_ntcIds):
    print(string_decorators, '\n', json.dumps(data, indent=4), '\n', string_decorators)