import pandas as pd


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add common technical indicators to a DataFrame with columns ['Open','High','Low','Close','Volume'].

    Adds:
    - SMA (7,30)
    - EMA (12,26)
    - MACD and MACD signal
    - RSI (14)
    - Bollinger Bands (20)
    - Daily returns

    Returns a copy with new columns.
    """
    df = df.copy()
    if 'Close' not in df.columns:
        raise ValueError('DataFrame must contain Close column')

    # Simple moving averages
    df['sma_7'] = df['Close'].rolling(window=7).mean()
    df['sma_30'] = df['Close'].rolling(window=30).mean()

    # Exponential moving averages
    df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['bb_upper'] = rolling_mean + (rolling_std * 2)
    df['bb_lower'] = rolling_mean - (rolling_std * 2)

    # Returns & volatility
    df['daily_return'] = df['Close'].pct_change()
    df['volatility_7'] = df['daily_return'].rolling(window=7).std()
    df['volatility_30'] = df['daily_return'].rolling(window=30).std()

    return df


def add_rolling_stats(df: pd.DataFrame, windows=(7, 30, 90)) -> pd.DataFrame:
    """Add rolling mean/std for Close and Volume for given windows."""
    df = df.copy()
    for w in windows:
        df[f'close_roll_mean_{w}'] = df['Close'].rolling(window=w).mean()
        df[f'close_roll_std_{w}'] = df['Close'].rolling(window=w).std()
        if 'Volume' in df.columns:
            df[f'volume_roll_mean_{w}'] = df['Volume'].rolling(window=w).mean()
            df[f'volume_roll_std_{w}'] = df['Volume'].rolling(window=w).std()
    return df
