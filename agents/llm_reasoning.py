"""
LLM Reasoning via LangChain.

Replaces raw ollama.chat() calls with LangChain chains for structured,
reproducible LLM reasoning. Uses ChatOllama by default (local Mistral),
but can be swapped to ChatOpenAI trivially.
"""

import json
import re
from typing import Optional

from agents.prompts import ANALYSIS_SYSTEM_PROMPT, ANALYSIS_USER_PROMPT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_llm(provider: str = "ollama", model: str = "mistral", temperature: float = 0.2):
    """Return a LangChain chat model instance.

    Args:
        provider: "ollama" or "openai"
        model: Model name (e.g. "mistral", "gpt-4o-mini")
        temperature: Sampling temperature
    """
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature)
    else:
        # Default: local Ollama
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(model=model, temperature=temperature)
        except ImportError:
            # Fallback for older installs
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(model=model, temperature=temperature)


def _parse_json_response(text: str) -> dict:
    """Extract and parse the first JSON object from LLM output."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try regex extraction
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


# ---------------------------------------------------------------------------
# Main reasoning function
# ---------------------------------------------------------------------------

def generate_rationale(
    ticker: str,
    portfolio_status: str,
    quote_data: str,
    xgboost_signal: str,
    xgboost_confidence: float,
    rag_context: str,
    provider: str = "ollama",
    model: str = "mistral",
) -> dict:
    """Generate a structured stock analysis using LangChain + LLM.

    Returns:
        dict with keys: recommendation, confidence, summary, reasons, risks
        Falls back to a basic dict on failure.
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = _get_llm(provider=provider, model=model)

    user_content = ANALYSIS_USER_PROMPT.format(
        ticker=ticker,
        portfolio_status=portfolio_status or "Not owned / not connected to Robinhood",
        quote_data=quote_data or "No real-time quote available",
        xgboost_signal=xgboost_signal or "No model prediction available",
        xgboost_confidence=xgboost_confidence if xgboost_confidence else 0,
        rag_context=rag_context or "No news context available",
    )

    messages = [
        SystemMessage(content=ANALYSIS_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    try:
        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        result = _parse_json_response(content)

        if result and "recommendation" in result:
            # Normalise recommendation value
            rec = result["recommendation"].upper().strip()
            if rec not in ("BUY", "SELL", "HOLD"):
                rec = "HOLD"
            result["recommendation"] = rec
            result.setdefault("confidence", 50)
            result.setdefault("summary", "Analysis complete.")
            result.setdefault("reasons", [])
            result.setdefault("risks", [])
            result["analysis_type"] = "langchain_llm"
            return result

        # If JSON parsing failed, try to extract recommendation from text
        return _fallback_from_text(content, ticker)

    except Exception as e:
        return {
            "recommendation": "HOLD",
            "confidence": 30,
            "summary": f"LLM reasoning unavailable: {str(e)}",
            "reasons": ["LLM service may be offline", "Using fallback assessment"],
            "risks": ["Analysis may be incomplete without LLM reasoning"],
            "analysis_type": "llm_error",
            "error": str(e),
        }


def _fallback_from_text(text: str, ticker: str) -> dict:
    """Extract a recommendation from free-form text when JSON parsing fails."""
    upper = text.upper()
    if "BUY" in upper and "SELL" not in upper:
        rec = "BUY"
    elif "SELL" in upper:
        rec = "SELL"
    else:
        rec = "HOLD"

    return {
        "recommendation": rec,
        "confidence": 40,
        "summary": text[:500] if text else "Could not parse LLM response.",
        "reasons": [],
        "risks": [],
        "analysis_type": "langchain_llm_text",
    }


# ---------------------------------------------------------------------------
# Quick check utility
# ---------------------------------------------------------------------------

def check_llm_available(provider: str = "ollama", model: str = "mistral") -> bool:
    """Check whether the LLM backend is reachable."""
    try:
        llm = _get_llm(provider=provider, model=model)
        from langchain_core.messages import HumanMessage
        resp = llm.invoke([HumanMessage(content="Say OK")])
        return bool(resp)
    except Exception:
        return False
