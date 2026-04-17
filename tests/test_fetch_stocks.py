import pandas as pd
from unittest import mock
from data import fetch_stocks


def test_fetch_stock_data_robust_monkeypatched():
    # 101 rows so the >100 guard in fetch_stock_data_robust passes
    fake_df = pd.DataFrame(
        {
            "Open": [100.0] * 101,
            "High": [101.0] * 101,
            "Low": [99.0] * 101,
            "Close": [100.5] * 101,
            "Volume": [1000] * 101,
            "Dividends": [0] * 101,
            "Stock Splits": [0] * 101,
        },
        index=pd.date_range("2024-01-01", periods=101),
    )

    class FakeTicker:
        def __init__(self, t):
            pass

        def history(self, period=None, **kwargs):
            return fake_df

        @property
        def info(self):
            return {"shortName": "Fake Corp", "marketCap": 123456789}

    # Patch the yf module that fetch_stocks imported at the top of that module
    with mock.patch.object(fetch_stocks.yf, "Ticker", FakeTicker):
        result, failed = fetch_stocks.fetch_stock_data_robust(
            ["FAKE"], period="1mo", max_retries=1
        )

    assert "FAKE" in result, "Expected FAKE in result dict"
    df = result["FAKE"]
    assert isinstance(df, pd.DataFrame)
    assert "Close" in df.columns
    assert not failed, f"Expected no failures, got: {failed}"
