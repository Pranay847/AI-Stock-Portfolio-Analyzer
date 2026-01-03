# Stock Portfolio Analyzer

Quickstart:

1. Create a virtualenv and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Add your OpenAI key:

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

3. Fetch raw data and process:

```bash
python data/fetch_stocks.py
python data/process_data.py
```

4. Train model (unified):

```bash
python models/train_xgboost.py
```

Or train per-ticker models:

```bash
python models/train_xgboost.py --per_ticker
```

5. Build FAISS index:

```bash
python rag/build_index.py
```

6. Run the Streamlit app:

```bash
streamlit run app.py
```

Files added/modified:
- `data/fetch_stocks.py`: saves dividends and metadata
- `data/process_data.py`: compute indicators and produce `data/processed/` CSVs
- `models/train_xgboost.py`: per-ticker or unified training, save predictions
- `models/predict.py`: helper for app predictions
- `rag/build_index.py`: enrich docs with metadata and news
- `app.py`: uses prediction helper and RAG utilities

Next suggestions: add scheduled runs (cron), add news scraping integration with a news API, and add backtesting modules.
