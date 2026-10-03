import numpy as np
import pandas as pd
import pytest

from data.features import add_rolling_stats, compute_technical_indicators


def _prices(closes):
    return pd.DataFrame(
        {"Close": closes, "Volume": [1000] * len(closes)},
        index=pd.date_range("2024-01-01", periods=len(closes)),
    )


def test_requires_close_column():
    with pytest.raises(ValueError):
        compute_technical_indicators(pd.DataFrame({"Open": [1.0, 2.0]}))


def test_does_not_mutate_input():
    df = _prices([float(i) for i in range(1, 41)])
    cols = list(df.columns)
    compute_technical_indicators(df)
    assert list(df.columns) == cols


def test_sma_and_warmup_nans():
    out = compute_technical_indicators(_prices([float(i) for i in range(1, 41)]))
    assert out["sma_7"].iloc[:6].isna().all()
    # mean of 1..7
    assert out["sma_7"].iloc[6] == pytest.approx(4.0)
    assert out["sma_30"].iloc[29] == pytest.approx(15.5)


def test_rsi_is_100_for_strictly_rising_prices():
    out = compute_technical_indicators(_prices([float(i) for i in range(1, 31)]))
    # avg_loss is 0, so rs is inf and RSI saturates at 100
    assert out["rsi_14"].iloc[:14].isna().all()
    assert out["rsi_14"].iloc[14:].eq(100.0).all()


def test_constant_prices_collapse_bands_and_macd():
    out = compute_technical_indicators(_prices([50.0] * 30))
    assert out["bb_upper"].iloc[-1] == pytest.approx(50.0)
    assert out["bb_lower"].iloc[-1] == pytest.approx(50.0)
    assert out["macd"].abs().max() == pytest.approx(0.0)
    assert np.allclose(out["daily_return"].iloc[1:], 0.0)


def test_add_rolling_stats_columns():
    out = add_rolling_stats(_prices([float(i) for i in range(1, 11)]), windows=(3,))
    assert out["close_roll_mean_3"].iloc[2] == pytest.approx(2.0)
    assert out["volume_roll_std_3"].iloc[-1] == pytest.approx(0.0)


def test_add_rolling_stats_without_volume():
    out = add_rolling_stats(pd.DataFrame({"Close": [1.0, 2.0, 3.0]}), windows=(2,))
    assert "close_roll_mean_2" in out.columns
    assert "volume_roll_mean_2" not in out.columns
