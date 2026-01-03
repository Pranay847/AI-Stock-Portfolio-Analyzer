import os
import glob
import joblib
import pandas as pd
import numpy as np


class Predictor:
    def __init__(self, model_dir='models/saved_models', unified_model_path=None, features=None):
        self.model_dir = model_dir
        self.unified_model_path = unified_model_path
        self.unified = None
        if unified_model_path and os.path.exists(unified_model_path):
            self.unified = joblib.load(unified_model_path)
        self.features = features

    def predict_for_ticker(self, ticker, X: pd.DataFrame):
        """Return (label, confidence, proba_vector)"""
        # try per-ticker model first
        model_path = os.path.join(self.model_dir, f"xgb_{ticker}.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        elif self.unified is not None:
            model = self.unified
        else:
            raise FileNotFoundError('No model available')

        Xc = X.fillna(0)
        proba = model.predict_proba(Xc)[0]
        label = int(model.predict(Xc)[0])
        return label, proba.max(), proba

    def predict_latest_for_all(self, processed_dir='data/processed'):
        files = glob.glob(os.path.join(processed_dir, '*.csv'))
        results = {}
        for f in files:
            try:
                df = pd.read_csv(f, index_col=0, parse_dates=True)
                ticker = os.path.splitext(os.path.basename(f))[0]
                latest = df.tail(1)
                if self.features is None:
                    # guess features
                    exclude = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'future_close', 'future_return', 'label', 'ticker']
                    features = [c for c in latest.columns if c not in exclude and latest[c].dtype in ["float64", "int64"]]
                else:
                    features = self.features
                X = latest[features]
                lbl, conf, proba = self.predict_for_ticker(ticker, X)
                results[ticker] = {'label': int(lbl), 'confidence': float(conf), 'proba': proba.tolist()}
            except Exception as e:
                results[ticker] = {'error': str(e)}
        return results


if __name__ == '__main__':
    p = Predictor(unified_model_path='models/saved_models/xgb_model.pkl')
    res = p.predict_latest_for_all()
    import json
    print(json.dumps(res, indent=2))
