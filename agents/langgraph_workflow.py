"""
LangGraph orchestration for the AI Stock Analyzer.

State machine:
  fetch_data → predict → retrieve_rag → llm_reasoning → recommend

Each node is a pure function that receives the current AnalysisState and
returns a partial dict that is merged into the state.
"""

import os
import pandas as pd
from typing import Any, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

from agents.llm_reasoning import generate_rationale
from agents.prompts import DEFAULT_EXPLAIN_PROMPT  # noqa: F401 (kept for importers)

# XGBoost label encoding: 0=SELL, 1=HOLD, 2=BUY
_LABEL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class AnalysisState(TypedDict):
    ticker: str
    portfolio_status: str          # human-readable portfolio context
    historical_json: str           # raw JSON from MCP fetch_historical_data
    quote_data: str                # formatted quote string
    xgboost_signal: str            # BUY / SELL / HOLD
    xgboost_confidence: float      # 0-100
    rag_context: str               # news + market context from vector DB
    result: dict                   # final recommendation dict


# ---------------------------------------------------------------------------
# Node: fetch_data
# ---------------------------------------------------------------------------

def fetch_data(state: AnalysisState) -> dict:
    """Fetch historical OHLCV and real-time quote via MCP tools."""
    ticker = state["ticker"]
    historical_json = ""
    quote_data = "No real-time quote available"

    try:
        from mcp_server import call_tool
        hist = call_tool("fetch_historical_data", ticker=ticker)
        historical_json = hist if isinstance(hist, str) else str(hist)

        quote_raw = call_tool("fetch_quote", ticker=ticker)
        if isinstance(quote_raw, dict):
            price = quote_raw.get("price", 0)
            change = quote_raw.get("change_percent", 0)
            volume = quote_raw.get("volume", "N/A")
            quote_data = (
                f"Price: ${price} | Change: {change}% | Volume: {volume}"
            )
        elif isinstance(quote_raw, str):
            quote_data = quote_raw
    except Exception as e:
        print(f"   ⚠️  MCP fetch_data error for {ticker}: {e}")

    return {"historical_json": historical_json, "quote_data": quote_data}


# ---------------------------------------------------------------------------
# Node: predict
# ---------------------------------------------------------------------------

def predict(state: AnalysisState) -> dict:
    """Run XGBoost prediction from pre-computed processed data."""
    ticker = state["ticker"]
    xgboost_signal = "No model prediction available"
    xgboost_confidence = 0.0

    try:
        from models.predict import Predictor
        processed_path = os.path.join("data", "processed", f"{ticker}.csv")
        if os.path.exists(processed_path):
            df = pd.read_csv(processed_path, index_col=0, parse_dates=True)
            exclude = {
                "Open", "High", "Low", "Close", "Adj Close", "Volume",
                "future_close", "future_return", "label", "ticker",
            }
            features = [
                c for c in df.columns
                if c not in exclude and df[c].dtype in ("float64", "int64")
            ]
            if features:
                X_latest = df[features].tail(1)
                predictor = Predictor()
                label, confidence, _ = predictor.predict_for_ticker(ticker, X_latest)
                xgboost_signal = _LABEL_MAP.get(int(label), "HOLD")
                xgboost_confidence = round(float(confidence) * 100, 1)
    except Exception as e:
        print(f"   ⚠️  XGBoost predict error for {ticker}: {e}")

    return {
        "xgboost_signal": xgboost_signal,
        "xgboost_confidence": xgboost_confidence,
    }


# ---------------------------------------------------------------------------
# Node: retrieve_rag
# ---------------------------------------------------------------------------

def retrieve_rag(state: AnalysisState) -> dict:
    """Retrieve semantic news + market context from Vector DB."""
    ticker = state["ticker"]
    rag_context = "No news context available"

    try:
        from alpha_vantage_vector_db import AlphaVantageVectorDB
        db = AlphaVantageVectorDB()
        ctx = db.get_stock_context(ticker)
        if ctx:
            rag_context = ctx
        else:
            # Attempt to fetch + store fresh data then retry
            result = db.process_stock(ticker, include_news=True)
            if result.get("success"):
                ctx = db.get_stock_context(ticker)
                if ctx:
                    rag_context = ctx
    except Exception as e:
        print(f"   ⚠️  Vector DB RAG error for {ticker}: {e}")

    return {"rag_context": rag_context}


# ---------------------------------------------------------------------------
# Node: llm_reasoning
# ---------------------------------------------------------------------------

def llm_reasoning(state: AnalysisState) -> dict:
    """Call LangChain reasoning chain to produce structured recommendation."""
    result = generate_rationale(
        ticker=state["ticker"],
        portfolio_status=state.get("portfolio_status") or "Not connected to Robinhood",
        quote_data=state.get("quote_data") or "No real-time quote available",
        xgboost_signal=state.get("xgboost_signal") or "No model prediction available",
        xgboost_confidence=state.get("xgboost_confidence") or 0,
        rag_context=state.get("rag_context") or "No news context available",
    )
    return {"result": result}


# ---------------------------------------------------------------------------
# Node: recommend
# ---------------------------------------------------------------------------

def recommend(state: AnalysisState) -> dict:
    """Attach metadata (ticker, xgb signal) to the final result dict."""
    result = dict(state.get("result") or {})
    result["symbol"] = state["ticker"]
    result["xgboost_signal"] = state.get("xgboost_signal", "N/A")
    result["xgboost_confidence"] = state.get("xgboost_confidence", 0)
    result.setdefault("analysis_type", "langgraph_workflow")
    return {"result": result}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    g = StateGraph(AnalysisState)
    g.add_node("fetch_data", fetch_data)
    g.add_node("predict", predict)
    g.add_node("retrieve_rag", retrieve_rag)
    g.add_node("llm_reasoning", llm_reasoning)
    g.add_node("recommend", recommend)

    g.set_entry_point("fetch_data")
    g.add_edge("fetch_data", "predict")
    g.add_edge("predict", "retrieve_rag")
    g.add_edge("retrieve_rag", "llm_reasoning")
    g.add_edge("llm_reasoning", "recommend")
    g.add_edge("recommend", END)

    return g.compile()


# Compiled graph — import and call run_analysis() from app code
_graph = None


def run_analysis(ticker: str, portfolio_status: str = "") -> dict:
    """Entry point: run the full LangGraph analysis pipeline for one ticker."""
    global _graph
    if _graph is None:
        _graph = build_graph()

    initial_state: AnalysisState = {
        "ticker": ticker.upper().strip(),
        "portfolio_status": portfolio_status,
        "historical_json": "",
        "quote_data": "",
        "xgboost_signal": "",
        "xgboost_confidence": 0.0,
        "rag_context": "",
        "result": {},
    }
    final_state = _graph.invoke(initial_state)
    return final_state.get("result", {})


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    result = run_analysis("AAPL")
    print(json.dumps(result, indent=2))
