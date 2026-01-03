import types
import pandas as pd
import os
from data import fetch_stocks


def test_fetch_stock_data_monkeypatched_tmp():
    # create a fake yfinance module with minimal Ticker behavior
    class FakeTicker:
        def __init__(self, ticker):
            self.ticker = ticker

        def history(self, period=None, interval=None, actions=None):
            # return a tiny dataframe resembling yfinance output
            return pd.DataFrame({
                'Open': [100.0, 101.0],
                'High': [101.0, 102.0],
                'Low': [99.0, 100.5],
                'Close': [100.5, 101.5],
                'Volume': [1000, 1100],
                'Dividends': [0, 0],
                'Stock Splits': [0, 0]
            }, index=pd.date_range('2025-01-01', periods=2))

        @property
        def info(self):
            return {'shortName': 'Fake Corp', 'marketCap': 123456789}

    fake_yf = types.SimpleNamespace(Ticker=lambda t: FakeTicker(t))

    # patch the _safe_import_yfinance function to return our fake module
    original_safe = fetch_stocks._safe_import_yfinance
    try:
        fetch_stocks._safe_import_yfinance = lambda: fake_yf

        res = fetch_stocks.fetch_stock_data(['FAKE'], period='1mo', interval='1d', retries=0, workers=1)
        assert 'FAKE' in res
        df = res['FAKE']
        assert isinstance(df, pd.DataFrame)
        assert 'Close' in df.columns
        # cleanup any files written
        csv = os.path.join('data', 'raw', 'FAKE.csv')
        meta = os.path.join('data', 'raw', 'FAKE_meta.json')
        if os.path.exists(csv):
            os.remove(csv)
        if os.path.exists(meta):
            os.remove(meta)
    finally:
        fetch_stocks._safe_import_yfinance = original_safe
