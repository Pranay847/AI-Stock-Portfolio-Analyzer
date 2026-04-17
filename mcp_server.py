"""
MCP Server — Tool Endpoints for the AI Stock Portfolio Analyzer.

Exposes 4 tools via Model Context Protocol (MCP):
  1. get_portfolio_status  – Check Robinhood ownership
  2. fetch_historical_data – Yahoo Finance OHLCV
  3. fetch_quote           – Alpha Vantage real-time quote
  4. fetch_news            – Alpha Vantage news + sentiment

The server uses stdio transport and is invoked in-process by the
LangGraph orchestrator.
"""

import os
import json
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# MCP FastMCP server
# ---------------------------------------------------------------------------
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("StockAnalyzerTools")


# ---------------------------------------------------------------------------
# Tool 1: Portfolio status from Robinhood
# ---------------------------------------------------------------------------
@mcp.tool()
def get_portfolio_status(ticker: str) -> str:
    """Check if a stock is owned in the Robinhood portfolio and return position details.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL)

    Returns:
        JSON string with ownership status and position details.
    """
    try:
        import robin_stocks.robinhood as rh

        positions = rh.account.get_open_stock_positions()
        for pos in positions:
            instrument_url = pos.get("instrument")
            instrument = rh.stocks.get_instrument_by_url(instrument_url)
            sym = instrument.get("symbol", "")
            if sym.upper() == ticker.upper():
                quote = rh.stocks.get_latest_price(sym)
                current_price = float(quote[0]) if quote and quote[0] else 0
                quantity = float(pos.get("quantity", 0))
                avg_buy = float(pos.get("average_buy_price", 0))
                equity = quantity * current_price
                cost_basis = quantity * avg_buy
                pl = equity - cost_basis
                pl_pct = ((current_price - avg_buy) / avg_buy * 100) if avg_buy > 0 else 0

                return json.dumps({
                    "owned": True,
                    "symbol": sym,
                    "name": instrument.get("simple_name", sym),
                    "quantity": quantity,
                    "average_buy_price": avg_buy,
                    "current_price": current_price,
                    "equity": equity,
                    "cost_basis": cost_basis,
                    "profit_loss": pl,
                    "profit_loss_percent": pl_pct,
                    "timestamp": datetime.now().isoformat(),
                })
        return json.dumps({"owned": False, "symbol": ticker.upper(), "timestamp": datetime.now().isoformat()})

    except Exception as e:
        return json.dumps({"owned": False, "symbol": ticker.upper(), "error": str(e),
                           "timestamp": datetime.now().isoformat()})


# ---------------------------------------------------------------------------
# Tool 2: Historical OHLCV from Yahoo Finance
# ---------------------------------------------------------------------------
@mcp.tool()
def fetch_historical_data(ticker: str, period: str = "6mo") -> str:
    """Fetch historical OHLCV data for a stock from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL)
        period: Time period — 1mo, 3mo, 6mo, 1y, 2y, 5y (default 6mo)

    Returns:
        JSON string with OHLCV rows (last 60 rows to keep payload small).
    """
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker.upper())
        df = stock.history(period=period)

        if df.empty:
            return json.dumps({"symbol": ticker.upper(), "error": "No data returned", "rows": []})

        # Keep last 60 rows for feature engineering
        df_recent = df.tail(60).copy()
        df_recent.index = df_recent.index.strftime("%Y-%m-%d")
        records = df_recent.reset_index().rename(columns={"index": "Date"}).to_dict(orient="records")

        return json.dumps({
            "symbol": ticker.upper(),
            "period": period,
            "total_rows": len(df),
            "returned_rows": len(records),
            "data": records,
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        return json.dumps({"symbol": ticker.upper(), "error": str(e), "rows": []})


# ---------------------------------------------------------------------------
# Tool 3: Real-time quote from Alpha Vantage
# ---------------------------------------------------------------------------
@mcp.tool()
def fetch_quote(ticker: str) -> str:
    """Fetch a real-time stock quote from Alpha Vantage.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL)

    Returns:
        JSON string with price, change, volume, etc.
    """
    import requests

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not api_key:
        return json.dumps({"symbol": ticker.upper(), "error": "ALPHA_VANTAGE_API_KEY not set"})

    try:
        url = "https://www.alphavantage.co/query"
        params = {"function": "GLOBAL_QUOTE", "symbol": ticker.upper(), "apikey": api_key}
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        gq = data.get("Global Quote", {})
        if not gq:
            return json.dumps({"symbol": ticker.upper(), "error": "No quote data"})

        return json.dumps({
            "symbol": gq.get("01. symbol", ticker.upper()),
            "price": float(gq.get("05. price", 0)),
            "open": float(gq.get("02. open", 0)),
            "high": float(gq.get("03. high", 0)),
            "low": float(gq.get("04. low", 0)),
            "volume": int(gq.get("06. volume", 0)),
            "previous_close": float(gq.get("08. previous close", 0)),
            "change": float(gq.get("09. change", 0)),
            "change_percent": gq.get("10. change percent", "0%").replace("%", ""),
            "latest_trading_day": gq.get("07. latest trading day", ""),
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        return json.dumps({"symbol": ticker.upper(), "error": str(e)})


# ---------------------------------------------------------------------------
# Tool 4: News + sentiment from Alpha Vantage
# ---------------------------------------------------------------------------
@mcp.tool()
def fetch_news(ticker: str, limit: int = 5) -> str:
    """Fetch news articles and sentiment scores for a stock from Alpha Vantage.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL)
        limit: Max number of articles (default 5)

    Returns:
        JSON string with list of news items including title, url, sentiment.
    """
    import requests

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not api_key:
        return json.dumps({"symbol": ticker.upper(), "error": "ALPHA_VANTAGE_API_KEY not set", "articles": []})

    try:
        url = "https://www.alphavantage.co/query"
        params = {"function": "NEWS_SENTIMENT", "tickers": ticker.upper(),
                  "limit": limit, "apikey": api_key}
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        feed = data.get("feed", [])
        if not feed:
            return json.dumps({"symbol": ticker.upper(), "articles": [],
                               "note": "No news available"})

        articles = []
        for item in feed[:limit]:
            ticker_sentiment = None
            for ts in item.get("ticker_sentiment", []):
                if ts.get("ticker", "").upper() == ticker.upper():
                    ticker_sentiment = ts
                    break

            articles.append({
                "title": item.get("title", ""),
                "summary": item.get("summary", "")[:300],
                "source": item.get("source", ""),
                "url": item.get("url", ""),
                "time_published": item.get("time_published", ""),
                "overall_sentiment": item.get("overall_sentiment_label", ""),
                "overall_sentiment_score": float(item.get("overall_sentiment_score", 0)),
                "ticker_sentiment": ticker_sentiment.get("ticker_sentiment_label", "") if ticker_sentiment else "",
                "ticker_sentiment_score": float(ticker_sentiment.get("ticker_sentiment_score", 0)) if ticker_sentiment else 0,
                "relevance_score": float(ticker_sentiment.get("relevance_score", 0)) if ticker_sentiment else 0,
            })

        return json.dumps({
            "symbol": ticker.upper(),
            "article_count": len(articles),
            "articles": articles,
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        return json.dumps({"symbol": ticker.upper(), "error": str(e), "articles": []})


# ---------------------------------------------------------------------------
# Direct-call helpers (used by LangGraph nodes without full MCP transport)
# ---------------------------------------------------------------------------

def call_tool(tool_name: str, **kwargs) -> dict:
    """Directly invoke an MCP tool and return parsed JSON result.

    This avoids the overhead of MCP transport for in-process use.
    """
    tool_map = {
        "get_portfolio_status": get_portfolio_status,
        "fetch_historical_data": fetch_historical_data,
        "fetch_quote": fetch_quote,
        "fetch_news": fetch_news,
    }
    fn = tool_map.get(tool_name)
    if fn is None:
        return {"error": f"Unknown tool: {tool_name}"}
    raw = fn(**kwargs)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Entry point (for standalone MCP server mode)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run()
