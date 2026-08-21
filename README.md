# Generative AI Stock Portfolio Analyzer

An AI-assisted portfolio analysis dashboard that combines market data, machine learning, retrieval-augmented generation, and natural-language explanations to help users understand stock positions and risk signals.

The project generates buy / sell / hold style insights with an XGBoost model, indexes financial context in vector databases for semantic retrieval, and presents portfolio-level analysis through a Streamlit interface.

## Features

- Fetches market data from Yahoo Finance and Alpha Vantage workflows.
- Builds technical indicators such as returns, moving averages, volatility, and trend features.
- Trains and runs XGBoost-based prediction workflows for stock signal generation.
- Uses vector databases for semantic retrieval over market and portfolio context.
- Produces LLM-assisted explanations for model outputs and portfolio observations.
- Supports portfolio-level analysis through Robinhood-oriented workflows.
- Provides a Streamlit UI for interactive analysis and visualization.
- Includes tests and standalone scripts for data, RAG, and provider-specific workflows.

## Tech Stack

- **App/UI:** Streamlit
- **Data:** yfinance, Alpha Vantage, Pandas, NumPy
- **Machine learning:** XGBoost, scikit-learn
- **RAG/vector search:** ChromaDB, vector stores, retrieval pipelines
- **LLM workflows:** LangGraph-style agents, prompt orchestration, LLM reasoning modules
- **Portfolio analysis:** Robinhood-oriented portfolio analyzer scripts
- **Testing:** pytest

## Architecture

```text
market and portfolio data
        |
        v
feature engineering + vector indexing
        |
        +--> XGBoost signal model
        |
        +--> RAG retrieval over financial context
        |
        v
LLM reasoning layer
        |
        v
Streamlit dashboard and analysis scripts
```

## Project Layout

```text
.
├── app.py
├── data/
│   ├── fetch_stocks.py
│   ├── features.py
│   └── process_data.py
├── models/
│   ├── train_xgboost.py
│   └── predict.py
├── rag/
│   ├── build_index.py
│   └── retrieval.py
├── agents/
│   ├── langgraph_workflow.py
│   ├── llm_reasoning.py
│   └── prompts.py
├── robinhood_portfolio_analyzer.py
├── alpha_vantage_vector_db.py
├── yahoo_finance_vector_db.py
└── tests/
```

## Local Development

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

Set any required API keys in `.env` before running provider-specific data or LLM workflows.

The app starts in **demo mode**, which hides the direct brokerage login and uses mock
portfolio data, so a hosted deployment never collects real credentials. To use the direct
Robinhood login form when running locally, set `DEMO_MODE=false` in `.env`.

### Connecting a real account safely

Set `SNAPTRADE_CLIENT_ID` and `SNAPTRADE_CONSUMER_KEY` (free keys from
[SnapTrade](https://snaptrade.com)) and visitors can connect their own Robinhood account
from the deployed app. The user authenticates on SnapTrade's hosted portal, so this app
never receives a password — it holds only a session-scoped access pair used to read
positions. This is the recommended path for any public deployment, since Robinhood has no
official OAuth and `robin_stocks` requires the account password directly.

LLM explanations resolve their backend automatically: OpenAI when `OPENAI_API_KEY` is set,
otherwise local Ollama. If neither is reachable, the app still runs and shows the XGBoost
signal without the narrative explanation.

## Performance

The multi-stock retrieval pipeline was optimized with concurrent fetching, batched
embeddings, and single-batch vector-store writes. The following figures were measured
on a 10-stock portfolio using the included `benchmark_retrieval.py` script:

| Stage | Before | After | Speedup |
|-------|--------|-------|---------|
| Fetch — sequential → concurrent (`ThreadPoolExecutor`) | 3.19 s | 0.71 s | ~4.5x |
| Embedding — per-item → single batched `encode()` | 374 ms | 196 ms | ~1.9x |
| Vector-store write — N upserts → 1 batched upsert | 1194 ms | 190 ms | ~6.3x |
| **End-to-end** (sum of the three stages) | **~4.8 s** | **~1.1 s** | **~4.3x** |

Reproduce on your own hardware:

```bash
pip install yfinance sentence-transformers chromadb
python benchmark_retrieval.py
```

Fetch timing depends on network conditions and Yahoo Finance rate limits, so the
concurrent-fetch speedup varies between runs; embedding and vector-store figures are
consistent. Numbers reflect a single machine and are meant as a reproducible baseline,
not a fixed guarantee.

## Engineering Highlights

- Built an end-to-end portfolio analysis workflow from data ingestion to model prediction and explanation.
- Combined XGBoost predictions with RAG-style retrieval so recommendations can include supporting context.
- Added separate vector database workflows for Yahoo Finance, Alpha Vantage, stock, and portfolio data.
- Organized LLM reasoning into agent, prompt, and retrieval modules instead of hard-coding explanations in the UI.
- Included test files for data-provider and portfolio-analysis workflows.
