"""Tests for services.stock_analysis and services.tickers.

The analyzer and the LangGraph workflow are replaced with fakes, so these run
offline and deterministically.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agents.langgraph_workflow as workflow
from services.stock_analysis import analyze_stock, price_targets
from services.tickers import resolve_ticker


class FakeVectorDB:
    def __init__(self, quote=None, raises=None):
        self._quote = quote
        self._raises = raises

    def fetch_quote(self, symbol):
        if self._raises:
            raise self._raises
        return self._quote


class FakeAnalyzer:
    def __init__(self, vector_db=None, analysis=None):
        self.vector_db = vector_db
        self._analysis = analysis or {
            'recommendation': 'BUY', 'confidence': 70,
            'summary': 'rule-based view', 'reasons': ['r1'], 'risks': ['k1'],
            'analysis_type': 'rule_based',
        }
        self.analyze_calls = 0

    def analyze_stock(self, symbol, position):
        self.analyze_calls += 1
        return dict(self._analysis)


QUOTE = {'price': 200.0, 'change_percent': '1.5', 'latest_trading_day': '2026-10-28',
         'market_state': 'CLOSED', 'source': 'yahoo'}


@pytest.fixture
def graph_fails(monkeypatch):
    """Make the LangGraph path fail so analysis falls back to the analyzer."""
    def boom(*a, **k):
        raise RuntimeError("no llm")
    monkeypatch.setattr(workflow, 'run_analysis', boom)
    monkeypatch.setattr(workflow, 'set_vector_db', lambda db: None)


# ---------------- ticker resolution ----------------

@pytest.mark.parametrize('query, expected', [
    ('AAPL', ('AAPL', True)),
    ('apple', ('AAPL', True)),
    ('Microsoft Corp', ('MSFT', True)),
    ('BRK-B', ('BRK-B', True)),
    ('', ('', False)),
])
def test_resolve_ticker(query, expected):
    assert resolve_ticker(query) == expected


@pytest.mark.parametrize('blank', ['  ', '\t', ' \n '])
def test_whitespace_does_not_resolve_to_a_real_ticker(blank):
    # "" is a substring of every company name, so the partial-match loop used to
    # resolve whitespace to the first entry in the table (AAPL).
    assert resolve_ticker(blank) == ('', False)


# ---------------- error paths never fabricate ----------------

def test_blank_query_is_an_error():
    r = analyze_stock('   ', FakeAnalyzer())
    assert r['data_fetch_failed'] is True
    assert r['symbol'] == 'INVALID'


def test_missing_analyzer_is_an_error_not_a_random_guess():
    results = [analyze_stock('AAPL', None) for _ in range(5)]
    for r in results:
        assert r['data_fetch_failed'] is True
        assert r['current_price'] == 0
        assert r['confidence'] == 0
        assert r['analysis_type'] == 'error'
    # The old fallback drew a random price and recommendation on every call.
    assert len({(r['current_price'], r['recommendation'], r['confidence']) for r in results}) == 1


def test_missing_market_data_client_names_the_missing_key():
    r = analyze_stock('AAPL', FakeAnalyzer(vector_db=None))
    assert r['data_fetch_failed'] is True
    assert 'not configured' in r['summary']
    assert any('ALPHA_VANTAGE_API_KEY' in reason for reason in r['reasons'])


def test_quote_unavailable_is_an_error():
    r = analyze_stock('AAPL', FakeAnalyzer(vector_db=FakeVectorDB(quote=None)))
    assert r['data_fetch_failed'] is True
    assert r['summary'] == 'Unable to fetch market data for AAPL.'


def test_unresolvable_name_says_so():
    r = analyze_stock('not a real company name', FakeAnalyzer(vector_db=FakeVectorDB(quote=None)))
    assert r['data_fetch_failed'] is True
    assert 'Could not resolve' in r['summary']


def test_exception_during_analysis_is_reported_not_fabricated():
    analyzer = FakeAnalyzer(vector_db=FakeVectorDB(raises=ConnectionError('upstream down')))
    r = analyze_stock('AAPL', analyzer)
    assert r['data_fetch_failed'] is True
    assert r['current_price'] == 0
    assert any('upstream down' in reason for reason in r['reasons'])


# ---------------- success paths ----------------

def test_falls_back_to_analyzer_when_graph_fails(graph_fails):
    analyzer = FakeAnalyzer(vector_db=FakeVectorDB(quote=QUOTE))
    r = analyze_stock('apple', analyzer)
    assert analyzer.analyze_calls == 1
    assert r.get('data_fetch_failed') is None
    assert r['symbol'] == 'AAPL'
    assert r['current_price'] == 200.0
    assert r['recommendation'] == 'BUY'
    assert r['ai_model'] == 'Rule-Based'
    assert (r['as_of'], r['market_state'], r['quote_source']) == ('2026-10-28', 'CLOSED', 'yahoo')
    assert (r['buy_price'], r['sell_price']) == price_targets('BUY', 70, 200.0)


def test_uses_graph_result_when_it_has_a_recommendation(monkeypatch):
    seen = {}

    def fake_run(symbol, portfolio_status, quote_data):
        seen.update(symbol=symbol, quote_data=quote_data)
        return {'recommendation': 'SELL', 'confidence': 80, 'summary': 'llm view',
                'analysis_type': 'langgraph_workflow', 'xgboost_signal': 'SELL',
                'xgboost_confidence': 0.71}

    monkeypatch.setattr(workflow, 'run_analysis', fake_run)
    monkeypatch.setattr(workflow, 'set_vector_db', lambda db: None)
    analyzer = FakeAnalyzer(vector_db=FakeVectorDB(quote=QUOTE))
    r = analyze_stock('AAPL', analyzer)

    assert analyzer.analyze_calls == 0
    assert r['recommendation'] == 'SELL'
    assert r['ai_model'] == 'LangChain + Mistral'
    assert r['holding_period'] == 'Short Term (1-3 months)'
    assert r['xgboost_signal'] == 'SELL'
    # The already-fetched quote is handed to the graph rather than re-fetched.
    assert seen['quote_data'].startswith('Price: $200.00')


def test_model_only_result_without_a_signal_falls_back(monkeypatch):
    monkeypatch.setattr(workflow, 'run_analysis', lambda *a, **k: {
        'recommendation': 'HOLD', 'analysis_type': 'model_only', 'xgboost_signal': None})
    monkeypatch.setattr(workflow, 'set_vector_db', lambda db: None)
    analyzer = FakeAnalyzer(vector_db=FakeVectorDB(quote=QUOTE))
    analyze_stock('AAPL', analyzer)
    assert analyzer.analyze_calls == 1


@pytest.mark.parametrize('rec', ['BUY', 'SELL', 'HOLD'])
def test_price_targets_bracket_the_price_sensibly(rec):
    buy, sell = price_targets(rec, 75, 100.0)
    assert buy < 100.0 < sell
