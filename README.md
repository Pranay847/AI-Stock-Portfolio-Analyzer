import yfinance as yf
import pandas as pd
import xgboost as xgb

def get_data(ticker, period="2y"):
    data = yf.download(ticker, period=period, interval="1d")
    data.dropna(inplace=True)
    return data
def add_features(df):
    df["return_1d"] = df["Adj Close"].pct_change()
    df["ma_10"] = df["Adj Close"].rolling(10).mean()
    df["ma_50"] = df["Adj Close"].rolling(50).mean()
    df["volatility_10"] = df["return_1d"].rolling(10).std()
    df.dropna(inplace=True)
    return df
def add_labels(df, horizon=5):
    df["future_return"] = df["Adj Close"].shift(-horizon) / df["Adj Close"] - 1
    df["target"] = (df["future_return"] > 0).astype(int)  # 1 = buy, 0 = sell/hold
    df.dropna(inplace=True)
    return df
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_model(df):
    features = ["return_1d", "ma_10", "ma_50", "volatility_10"]
    X = df[features]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    return model, features
    def generate_signals(df, model, features):
    probs = model.predict_proba(df[features])[:, 1]
    df["buy_prob"] = probs
    df["signal"] = df["buy_prob"].apply(lambda p: "BUY" if p > 0.6 else ("SELL" if p < 0.4 else "HOLD"))
    return df
from langchain_openai import ChatOpenAI

def explain_signals(ticker, latest_row):
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = f"""
    You are a financial assistant. For stock {ticker}, we have:

    - Current price: {latest_row['Adj Close']:.2f}
    - 1-day return: {latest_row['return_1d']:.4f}
    - MA10: {latest_row['ma_10']:.2f}
    - MA50: {latest_row['ma_50']:.2f}
    - Volatility (10d): {latest_row['volatility_10']:.4f}
    - Model signal: {latest_row['signal']} (buy_prob={latest_row['buy_prob']:.2f})

    Explain in simple terms why the model might be suggesting this action and provide a short, non-financial-advice style explanation.
    """

    return llm.invoke(prompt).content

