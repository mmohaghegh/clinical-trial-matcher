import streamlit as st
import json
from pathlib import Path

from clinical_trial_matcher.data.loader import EmbeddedData, DataLoader, EmbeddingResult, EmbeddingResults

MODEL_NAME = 'dmis-lab/biobert-base-cased-v1.1'
data_path = Path('./data')
file_path = data_path / "clinical_trials.json"

st.title("Clinical Trial Matcher Dashboard")

# Data loading
read_existing = st.checkbox("Read existing data", value=True)
store_data = st.checkbox("Store data after loading", value=True)

if read_existing:
    data_obj = DataLoader.from_file(file_path)
    st.success(f"Loaded {len(data_obj)} studies from existing data.")
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
    if store_data:
        data_obj.save_to_file(file_path)
    st.success(f"Loaded {len(data_obj)} studies from API.")


# Embedding
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

# Query input
query_text = st.text_input("Enter your query", "breast cancer treatment")
top_k = st.slider("Number of top matches", min_value=1, max_value=10, value=5)

if st.button("Find Matches"):
    top_indices = embedding_results.get_top_k_indices_based_on_similarity(query_text, k=top_k)
    top_ntcIds = data_obj.get_ntcIds_from_indices(top_indices)
    for data in data_obj.generate_entire_data_from_nctIds(top_ntcIds):
        st.markdown("---")
        st.json(data)