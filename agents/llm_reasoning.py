"""
LLM Reasoning via LangChain.

Replaces raw ollama.chat() calls with LangChain chains for structured,
reproducible LLM reasoning. The backend is resolved at call time: OpenAI when
OPENAI_API_KEY is set, otherwise local Ollama (Mistral). When neither is
reachable — e.g. a hosted deployment with no local Ollama daemon — reasoning
degrades to the model signal instead of raising.
"""

import json
import os
import re
from typing import Optional

from agents.prompts import ANALYSIS_SYSTEM_PROMPT, ANALYSIS_USER_PROMPT


# Each provider needs its own default model name, since callers pass a single
# `model` argument (or none at all).
#
# The Mistral default is deliberately a small model: on the free tier anything
# from mistral-small-latest upwards is refused with HTTP 429 and a
# x-ratelimit-limit-req-minute of 0, i.e. no quota at all. ministral-3b-latest
# is served there (750 req/min) and returns well-formed JSON for the analysis
# prompt. Workspaces with access to the larger models can override per provider
# via the environment -- see _MODEL_ENV_OVERRIDE.
_DEFAULT_MODELS = {
    "ollama": "mistral",
    "mistral": "ministral-3b-latest",
    "openai": "gpt-4o-mini",
}

# Per-provider model override, read at call time so it picks up values that
# arrive after import (Streamlit secrets, .env).
_MODEL_ENV_OVERRIDE = {
    "ollama": "OLLAMA_MODEL",
    "mistral": "MISTRAL_MODEL",
    "openai": "OPENAI_MODEL",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hosted_llm_configured() -> bool:
    """True when an API-key-based provider is available (needs no local server)."""
    return bool(os.getenv("MISTRAL_API_KEY") or os.getenv("OPENAI_API_KEY"))


def resolve_provider(provider: Optional[str] = None) -> str:
    """Choose an LLM backend.

    An explicit choice always wins. Otherwise prefer a hosted provider whenever
    its API key is present: hosted deployments (Streamlit Cloud and friends)
    cannot run a local Ollama server, so keying off the environment lets the
    same code serve both local development and production.
    """
    if provider and provider != "auto":
        return provider
    if os.getenv("MISTRAL_API_KEY"):
        return "mistral"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "ollama"


def _resolve_backend(provider: str = "auto", model: Optional[str] = None):
    """Resolve which LLM backend to use.

    Args:
        provider: "auto", "ollama", "mistral", or "openai"
        model: Model name, or None to use the provider's default

    Returns:
        Tuple of (provider, model)
    """
    provider = resolve_provider(provider)
    if not model:
        env_var = _MODEL_ENV_OVERRIDE.get(provider)
        model = (os.getenv(env_var) if env_var else None) or \
            _DEFAULT_MODELS.get(provider, "mistral")
    return provider, model


def _get_llm(provider: Optional[str] = None, model: Optional[str] = None,
             temperature: float = 0.2):
    """Return a LangChain chat model instance.

    Args:
        provider: "ollama", "mistral" or "openai". None auto-detects.
        model: Model name. None uses the provider's default.
        temperature: Sampling temperature
    """
    provider = resolve_provider(provider)
    model = model or _DEFAULT_MODELS.get(provider, "mistral")

    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model=model, temperature=temperature)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature)

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
    provider: str = "auto",
    model: Optional[str] = None,
) -> dict:
    """Generate a structured stock analysis using LangChain + LLM.

    Returns:
        dict with keys: recommendation, confidence, summary, reasons, risks
        Falls back to the raw model signal when no LLM backend is reachable.
    """
    provider, model = _resolve_backend(provider, model)

    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        llm = _get_llm(provider=provider, model=model)
    except Exception as e:
        return _model_only_result(xgboost_signal, xgboost_confidence, e)

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
        return _model_only_result(xgboost_signal, xgboost_confidence, e)


def _model_only_result(xgboost_signal: str, xgboost_confidence: float, error: Exception) -> dict:
    """Result used when no LLM backend is reachable.

    Surfaces the XGBoost signal on its own rather than failing the analysis, so
    a deployment without an LLM still shows the model's prediction.
    """
    rec = (xgboost_signal or "HOLD").upper().strip()
    if rec not in ("BUY", "SELL", "HOLD"):
        rec = "HOLD"

    confidence = xgboost_confidence or 0
    if 0 < confidence <= 1:
        confidence *= 100

    return {
        "recommendation": rec,
        "confidence": int(confidence) or 30,
        "summary": (
            "Model signal only — no LLM backend is configured, so the numeric "
            "prediction is shown without a narrative explanation."
        ),
        "reasons": [f"XGBoost signal: {rec}"],
        "risks": ["Explanation layer unavailable; the signal is not interpreted."],
        "analysis_type": "model_only",
        "error": str(error),
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

def check_llm_available(provider: str = "auto", model: Optional[str] = None) -> bool:
    """Check whether the LLM backend is reachable."""
    try:
        provider, model = _resolve_backend(provider, model)
        llm = _get_llm(provider=provider, model=model)
        from langchain_core.messages import HumanMessage
        resp = llm.invoke([HumanMessage(content="Say OK")])
        return bool(resp)
    except Exception:
        return False
