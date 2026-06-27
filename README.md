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

## Engineering Highlights

- Built an end-to-end portfolio analysis workflow from data ingestion to model prediction and explanation.
- Combined XGBoost predictions with RAG-style retrieval so recommendations can include supporting context.
- Added separate vector database workflows for Yahoo Finance, Alpha Vantage, stock, and portfolio data.
- Organized LLM reasoning into agent, prompt, and retrieval modules instead of hard-coding explanations in the UI.
- Included test files for data-provider and portfolio-analysis workflows.

## Resume Highlights

- Built an AI stock portfolio analyzer using Python, Streamlit, Pandas, XGBoost, ChromaDB, RAG, and LLM reasoning.
- Indexed financial and portfolio data in vector databases to support semantic retrieval for stock analysis.
- Generated portfolio insights and buy / sell / hold style signals by combining technical indicators, ML predictions, and natural-language explanations.
