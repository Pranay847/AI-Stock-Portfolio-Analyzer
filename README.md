# Generative AI Stock Portfolio Analyzer

An AI-assisted portfolio analysis dashboard that combines market data, machine learning, retrieval-augmented generation, and natural-language explanations to help users understand stock positions and risk signals.

The project generates buy / sell / hold style insights with an XGBoost model, indexes financial context in vector databases for semantic retrieval, and presents portfolio-level analysis through a React + TypeScript web app backed by a FastAPI service.

## Features

- Fetches market data from Yahoo Finance and Alpha Vantage workflows.
- Builds technical indicators such as returns, moving averages, volatility, and trend features.
- Trains and runs XGBoost-based prediction workflows for stock signal generation.
- Uses vector databases for semantic retrieval over market and portfolio context.
- Produces LLM-assisted explanations for model outputs and portfolio observations.
- Supports portfolio-level analysis through Robinhood-oriented workflows.
- Provides a React + TypeScript UI (with a legacy Streamlit UI kept during the migration).
- Connects a real brokerage account through SnapTrade without the app ever seeing a password.
- Includes tests and standalone scripts for data, RAG, and provider-specific workflows.

## Tech Stack

- **Frontend:** React 19, TypeScript, Vite, TanStack Query, Tailwind CSS
- **API:** FastAPI, server-side sessions, background analysis jobs
- **Legacy UI:** Streamlit
- **Data:** yfinance, Alpha Vantage, Pandas, NumPy
- **Machine learning:** XGBoost, scikit-learn
- **RAG/vector search:** ChromaDB, vector stores, retrieval pipelines
- **LLM workflows:** LangGraph-style agents, prompt orchestration, LLM reasoning modules
- **Portfolio analysis:** Robinhood-oriented portfolio analyzer scripts
- **Testing:** pytest, Vitest, Testing Library
- **Deployment:** single Docker image (frontend + API), Render Blueprint

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
FastAPI  <-->  React + TypeScript web app
```

## Project Layout

```text
.
├── web/                         React + TypeScript frontend (Vite)
│   └── src/{screens,components,api,lib}
├── api/                         FastAPI app: JSON under /api, serves web/dist at /
│   ├── main.py                  routes, session cookie, SPA serving
│   ├── sessions.py              server-side sessions
│   └── jobs.py                  background portfolio analysis
├── services/                    framework-free logic shared by api/ and app.py
│   ├── stock_analysis.py
│   ├── tickers.py
│   └── owner_access.py          gate for personal SnapTrade keys
├── app.py                       legacy Streamlit UI (kept until the React app reaches parity)
├── agents/                      LangGraph workflow, LLM reasoning, prompts
├── models/                      XGBoost training and prediction
├── data/                        price download and feature engineering
├── robinhood_portfolio_analyzer.py
├── alpha_vantage_vector_db.py   quotes, news and RAG context (ChromaDB)
├── snaptrade_connector.py       brokerage connection via SnapTrade
├── Dockerfile                   one image: built frontend + API
├── render.yaml                  Render Blueprint for that image
└── tests/
```

## Local Development

### React frontend + API (the new app)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env              # then fill in keys

uvicorn api.main:app --reload     # API on :8000

cd web && npm install && npm run dev   # UI on :5173, proxying /api to :8000
```

Or run the whole thing as it ships, in one container:

```bash
docker build -t ai-stock-analyzer .
docker run -p 8000:8000 --env-file .env ai-stock-analyzer   # http://localhost:8000
```

The image serves the built frontend and the API from one origin, so it deploys to any
container host (Render, Railway, Fly.io). The Python side cannot run as serverless
functions: ChromaDB needs a persistent process and portfolio analysis runs for minutes.

### Deploy to Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Pranay847/AI-Stock-Portfolio-Analyzer)

`render.yaml` defines the service as a Render Blueprint: the Docker image above, a health
check on `/api/health`, and a redeploy on every push to the default branch.

1. Click the button and sign in to Render with GitHub.
2. Render asks for `ALPHA_VANTAGE_API_KEY`, `MISTRAL_API_KEY`, `SNAPTRADE_CLIENT_ID` and
   `SNAPTRADE_CONSUMER_KEY`. They are stored in Render's environment settings, never in the
   repository.
3. Deploy the Blueprint. The first build takes several minutes.
4. Render generates `OWNER_ACCESS_TOKEN`. Copy it from the service's **Environment** tab and
   enter it on the My Portfolio screen to connect your brokerage account.

On the free plan the service sleeps after 15 idle minutes, and the next visit waits about a
minute while it starts. Sessions are kept in memory, so a restart, redeploy or sleep signs
everyone out and the owner token has to be entered again. The free plan's 0.1 CPU also makes
analysis slow: the four-position demo analysis took 52 s in the image run under the same
limits. The `starter` plan stays awake and has more CPU; change `plan:` in `render.yaml`.

### Legacy Streamlit app

```bash
streamlit run app.py
```

### Tests

```bash
pytest                       # Python: services, API, contract
cd web && npm test           # frontend
```

Set any required API keys in `.env` before running provider-specific data or LLM workflows.

The app starts in **demo mode**, which hides the direct brokerage login and uses mock
portfolio data, so a hosted deployment never collects real credentials. To use the direct
Robinhood login form when running locally, set `DEMO_MODE=false` in `.env`.

### Connecting a real account safely

Set `SNAPTRADE_CLIENT_ID` and `SNAPTRADE_CONSUMER_KEY` from
[SnapTrade](https://snaptrade.com) and users can connect a Robinhood account from the
deployed app. They authenticate on SnapTrade's hosted portal, so this app never receives a
password — it holds only a session-scoped access pair used to read positions. This is the
recommended path for any public deployment, since Robinhood has no official OAuth and
`robin_stocks` requires the account password directly.

SnapTrade issues two kinds of key, selected with `SNAPTRADE_AUTH_MODE`:

| Mode | Key type | Who can connect |
|------|----------|-----------------|
| `personal` (default) | Free, one per person | Only your own brokerage accounts |
| `commercial` | Paid / approval-gated | Any visitor connects their own account |

The default is `personal`, so a free SnapTrade key works against your own portfolio. Because
a personal key always resolves to its owner's account, connecting is **disabled until you set
`OWNER_ACCESS_TOKEN`** (16+ characters), and each session must enter that token before it can
connect -- otherwise any visitor to a public deployment could load your holdings. A deployment
where strangers connect their own accounts needs a `commercial` key and
`SNAPTRADE_AUTH_MODE=commercial`, which requires no owner token.

LLM explanations resolve their backend automatically: Mistral when `MISTRAL_API_KEY` is set,
then OpenAI when `OPENAI_API_KEY` is set, otherwise local Ollama. If neither is reachable, the app still runs and shows the XGBoost
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
pip install yfinance chromadb
python benchmark_retrieval.py
```

Fetch timing depends on network conditions and Yahoo Finance rate limits, so the
concurrent-fetch speedup varies between runs; embedding and vector-store figures are
consistent. Numbers reflect a single machine and are meant as a reproducible baseline,
not a fixed guarantee.

The deployed app embeds differently from the benchmark's batched path: one document per
call, on a single-threaded ONNX session. Measured in the Docker image under Render's
free-plan limits (0.1 CPU, 512 MB), the single-threaded session embedded 10 news items in
9.4 s against 20 s for ChromaDB's default session, and batching made little difference to it
(10.3 s). Batching did cost memory: batched calls from concurrent analyses took the process
past 512 MB and it was killed, while per-document calls kept the same workload at a 358 MB
peak. Vector-store writes are still batched.

## Engineering Highlights

- Built an end-to-end portfolio analysis workflow from data ingestion to model prediction and explanation.
- Combined XGBoost predictions with RAG-style retrieval so recommendations can include supporting context.
- Added separate vector database workflows for Yahoo Finance, Alpha Vantage, stock, and portfolio data.
- Organized LLM reasoning into agent, prompt, and retrieval modules instead of hard-coding explanations in the UI.
- Included test files for data-provider and portfolio-analysis workflows.
