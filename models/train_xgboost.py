import os
import glob
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
from data.features import compute_technical_indicators, add_rolling_stats


def load_dataset(processed_dir):
    files = glob.glob(os.path.join(processed_dir, "*.csv"))
    dfs = []
    for f in files:
        df = pd.read_csv(f, parse_dates=True, index_col=0)
        ticker = os.path.splitext(os.path.basename(f))[0]
        df['ticker'] = ticker
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f'No CSVs found in {processed_dir}')
    data = pd.concat(dfs, axis=0)
    return data


def create_labels(df, horizon=1, threshold=0.02):
    """Label data: future pct change > threshold => buy (2), < -threshold => sell (0), else hold (1)"""
    df = df.copy()
    df['future_close'] = df.groupby('ticker')['Close'].shift(-horizon)
    df['future_return'] = (df['future_close'] - df['Close']) / df['Close']
    df['label'] = 1
    df.loc[df['future_return'] > threshold, 'label'] = 2
    df.loc[df['future_return'] < -threshold, 'label'] = 0
    df = df.dropna(subset=['future_return'])
    return df


def train_model(data, features, output_path):
    X = data[features].fillna(0)
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(clf, output_path)
    print(f"Saved model to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_dir', default='data/processed', help='Processed CSV directory')
    parser.add_argument('--output', default='models/saved_models/xgb_model.pkl')
    parser.add_argument('--per_ticker', action='store_true', help='Train a separate model per ticker')
    args = parser.parse_args()

    data = load_dataset(args.processed_dir)
    # Ensure indicators exist
    data = data.groupby('ticker').apply(lambda d: compute_technical_indicators(d)).reset_index(drop=True)
    data = add_rolling_stats(data)
    data = create_labels(data)

    # Select feature columns automatically
    exclude = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'future_close', 'future_return', 'label', 'ticker']
    features = [c for c in data.columns if c not in exclude and data[c].dtype in ["float64", "int64"]]
    print(f"Training with {len(features)} features")
    if args.per_ticker:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        preds_dir = os.path.join('models', 'predictions')
        os.makedirs(preds_dir, exist_ok=True)
        for ticker, grp in data.groupby('ticker'):
            try:
                X = grp[features].fillna(0)
                y = grp['label']
                if len(y.unique()) < 2:
                    print(f"Skipping {ticker}: not enough label variety")
                    continue
                clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
                clf.fit(X, y)
                model_path = os.path.join('models', 'saved_models', f"xgb_{ticker}.pkl")
                joblib.dump(clf, model_path)
                # save prediction probabilities for latest rows
                probs = clf.predict_proba(X.tail(30).fillna(0))
                out = grp.tail(30).copy()
                out[['p_sell','p_hold','p_buy']] = probs
                out.to_csv(os.path.join(preds_dir, f"{ticker}_preds.csv"))
                print(f"Trained and saved {ticker} model -> {model_path}")
            except Exception as e:
                print(f"Error training {ticker}: {e}")
    else:
        train_model(data, features, args.output)
        # save unified recent predictions
        model = joblib.load(args.output)
        preds_dir = os.path.join('models', 'predictions')
        os.makedirs(preds_dir, exist_ok=True)
        X_recent = data.groupby('ticker').tail(1)[features].fillna(0)
        if not X_recent.empty:
            probs = model.predict_proba(X_recent)
            out = data.groupby('ticker').tail(1).copy()
            out[['p_sell','p_hold','p_buy']] = probs
            out.to_csv(os.path.join(preds_dir, f"unified_recent_preds.csv"))
            print(f"Saved unified recent predictions -> {os.path.join(preds_dir, 'unified_recent_preds.csv')}")
