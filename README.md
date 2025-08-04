# Clinical Trial Matcher

Clinical Trial Matcher is a tool for searching and matching clinical trials based on natural language queries. It leverages BioBERT embeddings to find the most relevant clinical trials for a given query.

## Features

- Fetch clinical trial data from [clinicaltrials.gov](https://clinicaltrials.gov/)
- Store and reload trial data locally
- Embed trial descriptions, summaries, and eligibility criteria using BioBERT
- Find top matching trials for a query using semantic similarity
- Interactive dashboard using Streamlit

## Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd clinical-trial-matcher
   ```
2. Install dependencies:
   ```bash
   poetry install
   ```

## Usage

### Initial Data Loading

Run the script to fetch and embed clinical trial data:
```bash
python scripts/initial_data_loading.py
```

### Dashboard

Launch the Streamlit dashboard:
```bash
streamlit run scripts/dashboard.py
```

## Configuration

- Change the model name in scripts if you want to use a different BERT variant.
- Data is stored in the `data/clinical_trials.json` file by default.

## Project Structure

- `clinical_trial_matcher/data/loader.py`: Data loading and embedding logic
- `clinical_trial_matcher/model/embeddings.py`: BioBERT embedding utilities
- `scripts/initial_data_loading.py`: Script for initial data fetch and embedding
- `scripts/dashboard.py`: Streamlit dashboard for interactive trial matching

## Requirements

- Python 3.10+
- PyTorch
- Transformers
- Scikit-learn
- Streamlit
- tqdm
- requests
- numpy

## License

MIT License

## Acknowledgements

- [BioBERT](https://github.com/dmis-lab/biobert)
- [clinicaltrials.gov API](https://clinicaltrials.gov/api/gui)
