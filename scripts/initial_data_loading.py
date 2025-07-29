import json
import requests
from pathlib import Path

from tqdm import tqdm

from clinical_trial_matcher.data.loader import EmbeddedData, DataLoader

READ_EXISTING_DATA = True
data_path = Path('./data')
file_path = data_path / "clinical_trials.json"

if READ_EXISTING_DATA:
    with open(file_path, "r") as f:
        studies = json.load(f)
    print(f"Loaded {len(studies)} studies from existing data.")

else:
    params = {
        "format": "json",
        "query.cond": "cancer",
        "pageSize": 10000,
    }
    studies = []
    url = "https://clinicaltrials.gov/api/v2/studies"
    r = requests.get(url, params=params)
    data = r.json()
    studies.extend(data["studies"])
    for i in tqdm(range(10), desc="Loading studies page by page"):
        params["pageToken"] = data["nextPageToken"]
        r = requests.get(url, params=params)
        data = r.json()
        studies.extend(data["studies"])

    with open(file_path, "w") as f:
        json.dump(studies, f)

data_obj = DataLoader(studies)
embedding_data = EmbeddedData(data_obj)
embedding_data.embed_for_key(['protocolSection', 'identificationModule', 'briefTitle'])
print(studies[0]['protocolSection']['identificationModule']['briefTitle'])
print(studies[0]['protocolSection']['descriptionModule']['briefSummary'])
print(studies[0]['protocolSection']['eligibilityModule']['eligibilityCriteria'])
print(studies[0]['protocolSection']['conditionsModule']['conditions'])