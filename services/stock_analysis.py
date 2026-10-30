"""Single-stock analysis, independent of any UI framework.

Both the Streamlit app and the HTTP API call analyze_stock(). It returns a
result dict on success and an error dict on failure -- marked with
data_fetch_failed=True -- and never fabricates a price or recommendation.
"""

import logging
from datetime import datetime

from services.tickers import resolve_ticker

logger = logging.getLogger(__name__)

# analysis_type values that mean a LangChain/LangGraph LLM produced the call.
_LLM_ANALYSIS_TYPES = ('langchain_llm', 'langchain_llm_text', 'langgraph_workflow')


def price_targets(recommendation: str, confidence: float, current_price: float) -> tuple[float, float]:
    """Entry (buy) and exit (sell) targets scaled by recommendation and confidence."""
    if recommendation == 'BUY':
        # Higher confidence = more aggressive upside target; entry slightly
        # below current, tighter with higher confidence.
        sell_price = current_price * (1 + 0.08 + confidence * 0.0015)
        buy_price = current_price * (1 - 0.03 + confidence * 0.0002)
    elif recommendation == 'SELL':
        # Higher confidence = more aggressive downside protection; exit target
        # above current, closer with higher confidence.
        buy_price = current_price * (1 - 0.10 - confidence * 0.001)
        sell_price = current_price * (1 + 0.02 - confidence * 0.0001)
    else:  # HOLD: moderate targets for range-bound trading
        buy_price = current_price * (1 - 0.05)
        sell_price = current_price * (1 + 0.08 + confidence * 0.0005)
    return buy_price, sell_price


def _error(symbol: str, summary: str, reasons: list[str], ai_model: str) -> dict:
    return {
        'symbol': symbol,
        'current_price': 0,
        'buy_price': 0,
        'sell_price': 0,
        'recommendation': 'HOLD',
        'confidence': 0,
        'holding_period': 'N/A',
        'summary': summary,
        'reasons': reasons,
        'analysis_type': 'error',
        'ai_model': ai_model,
        'data_fetch_failed': True,
    }


def _ai_model_label(analysis_type: str) -> str:
    if analysis_type in _LLM_ANALYSIS_TYPES:
        return 'LangChain + Mistral'
    if analysis_type == 'mistral_ai':
        return 'Mistral AI'
    return 'Rule-Based'


def analyze_stock(query: str, analyzer) -> dict:
    """Analyze one stock.

    Args:
        query: A ticker or company name; resolved via services.tickers.
        analyzer: A RobinhoodPortfolioAnalyzer, or None if it failed to
            initialise -- which yields an error result, not a guess.
    """
    original_input = query
    symbol, is_valid = resolve_ticker(query)

    if not symbol:
        return _error(
            'INVALID',
            'Please enter a stock symbol or company name',
            ['Example: "AAPL" or "Apple", "MSFT" or "Microsoft"'],
            'Error',
        )

    if analyzer is None:
        return _error(
            symbol,
            'The analysis engine is unavailable, so no analysis was run.',
            ['The portfolio analyzer failed to initialise; check the server logs'],
            'Unavailable',
        )

    try:
        # Get stock quote only (1 API call instead of 2-3)
        current_price = 0
        change_percent = 0
        as_of = ''
        market_state = ''
        quote_source = ''

        if analyzer.vector_db:
            quote = analyzer.vector_db.fetch_quote(symbol)
            if quote:
                current_price = quote.get('price', 0)
                change_percent = float(quote.get('change_percent', 0))
                # Alpha Vantage's free quote is the last *completed* session,
                # not an intraday price. Carry the date through so the UI can
                # say so rather than implying the number is live.
                as_of = quote.get('latest_trading_day', '')
                market_state = quote.get('market_state', '')
                quote_source = quote.get('source', '')

        position = {
            'symbol': symbol,
            'name': symbol,
            'quantity': 0,
            'average_buy_price': current_price,
            'current_price': current_price,
            'equity': 0,
            'cost_basis': 0,
            'profit_loss': 0,
            'profit_loss_percent': change_percent,
            'timestamp': datetime.now().isoformat(),
        }

        # Only analyze when we actually have a quote. Reasoning about a stock
        # whose price never arrived spends an LLM round trip (and, via the
        # graph, more provider calls) to produce commentary about missing data
        # that the error path below discards anyway.
        analysis = {}
        if current_price > 0:
            # Try the LangGraph workflow first, fall back to the analyzer
            try:
                from agents.langgraph_workflow import run_analysis, set_vector_db

                # Donate the analyzer's vector DB so the graph does not build a
                # second one at a different path with a cold cache, which
                # re-fetches the quote, overview and news we already paid for.
                set_vector_db(analyzer.vector_db)

                portfolio_status = (
                    f"Owned: {'Yes' if position['quantity'] > 0 else 'No'} | "
                    f"Price: ${current_price:.2f} | Change: {change_percent:.2f}%"
                )
                # Hand over the quote we already fetched. Without this the
                # graph fetches its own, doubling the provider calls per lookup
                # and -- when that second call fails -- narrating "Price: $0"
                # while the metrics on screen show the real one.
                quote_summary = (
                    f"Price: ${current_price:.2f} | "
                    f"Change: {change_percent:.2f}%"
                )
                analysis = run_analysis(
                    symbol,
                    portfolio_status=portfolio_status,
                    quote_data=quote_summary,
                )
                # With no LLM reachable, reasoning degrades to a model-only
                # result. That is worth showing when XGBoost actually predicted
                # something, but a bare HOLD with no model behind it carries no
                # information -- fall through in that case.
                model_only_without_signal = (
                    analysis.get("analysis_type") == "model_only"
                    and analysis.get("xgboost_signal") not in ("BUY", "SELL", "HOLD")
                )
                if not analysis.get("recommendation") or model_only_without_signal:
                    raise ValueError("langgraph produced no usable result")
            except Exception:
                analysis = analyzer.analyze_stock(symbol, position)

        if current_price == 0:
            if not is_valid:
                return _error(
                    symbol,
                    f'Could not resolve "{original_input}" to a valid ticker symbol.',
                    [
                        f'Unknown company or ticker: {original_input}',
                        'Try using the exact ticker symbol (e.g., MSFT, AAPL)',
                        'Or check the spelling of the company name',
                    ],
                    'Data Unavailable',
                )
            if analyzer.vector_db is None:
                # The market-data client failed to initialise, which in
                # practice means ALPHA_VANTAGE_API_KEY is missing. Say so: a
                # message blaming rate limits sends people looking for a broken
                # API when nothing had been called at all.
                return _error(
                    symbol,
                    'Market data is not configured, so no quote could be requested.',
                    [
                        'ALPHA_VANTAGE_API_KEY is not set for this deployment',
                        'Set it as an environment variable (or in Streamlit secrets)',
                        'Locally: add it to your .env file',
                        'Get a free key at https://www.alphavantage.co/support/#api-key',
                    ],
                    'Data Unavailable',
                )
            return _error(
                symbol,
                f'Unable to fetch market data for {symbol}.',
                [
                    'Real-time data unavailable',
                    'Possible reasons: API rate limits, invalid ticker, or network issues',
                    'Alpha Vantage free tier: 25 API calls/day limit',
                ],
                'Data Unavailable',
            )

        recommendation = analysis.get('recommendation', 'HOLD')
        confidence = analysis.get('confidence', 50)
        buy_price, sell_price = price_targets(recommendation, confidence, current_price)
        analysis_type = analysis.get('analysis_type', 'ai')

        return {
            'symbol': symbol,
            'current_price': current_price,
            'buy_price': buy_price,
            'sell_price': sell_price,
            'recommendation': recommendation,
            'confidence': confidence,
            'holding_period': (
                'Short Term (1-3 months)' if recommendation == 'SELL' else 'Long Term (6+ months)'
            ),
            'summary': analysis.get('summary', 'Analysis complete.'),
            'reasons': analysis.get('reasons', []),
            'risks': analysis.get('risks', []),
            'xgboost_signal': analysis.get('xgboost_signal'),
            'xgboost_confidence': analysis.get('xgboost_confidence'),
            'analysis_type': analysis_type,
            'ai_model': _ai_model_label(analysis_type),
            'as_of': as_of,
            'market_state': market_state,
            'quote_source': quote_source,
        }
    except Exception as e:
        # Previously this fell through to a randomly generated price and
        # BUY/SELL/HOLD call. A fabricated recommendation on a made-up price is
        # worse than no answer for a financial tool, so report the failure.
        logger.exception("analysis failed for %s", symbol)
        return _error(
            symbol,
            f'Analysis failed for {symbol}.',
            [f'{type(e).__name__}: {e}', 'Try again in a moment'],
            'Error',
        )
