import os
import glob
import json
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

import pandas as pd
import yfinance as yf

load_dotenv()


def build_index_from_raw(raw_dir='data/raw', index_dir='rag/index'):
    os.makedirs(index_dir, exist_ok=True)
    files = glob.glob(os.path.join(raw_dir, '*.csv'))
    texts = []
    metadatas = []

    for f in files:
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            ticker = os.path.splitext(os.path.basename(f))[0]
            latest = df['Close'].iloc[-1]
            mean = df['Close'].mean()
            vol = df['Close'].pct_change().std()
            text = f"Ticker: {ticker}\nLatest Close: {latest:.2f}\nMean Close: {mean:.2f}\nVolatility: {vol:.4f}\nRows: {len(df)}\n"
            # include recent 30-day summary
            recent = df['Close'].tail(30).describe().to_dict()
            text += "Recent30: " + json.dumps(recent)

            # include metadata if available
            meta_path = os.path.join(raw_dir, f"{ticker}_meta.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as fh:
                        meta = json.load(fh)
                    for k, v in meta.items():
                        text += f"\n{k}: {v}"
                except Exception:
                    pass

            # include recent news if provided by yfinance
            try:
                t = yf.Ticker(ticker)
                news = getattr(t, 'news', None)
                if callable(news):
                    nitems = t.news()
                else:
                    nitems = t.news if hasattr(t, 'news') else []
                if nitems:
                    snippets = []
                    for ni in nitems[:5]:
                        title = ni.get('title') if isinstance(ni, dict) else None
                        if title:
                            snippets.append(title)
                    if snippets:
                        text += "\nNewsTitles: " + json.dumps(snippets)
            except Exception:
                pass

            # include model predictions if available
            preds_path = os.path.join('models', 'predictions')
            unified_preds = os.path.join(preds_path, 'unified_recent_preds.csv')
            per_path = os.path.join(preds_path, f"{ticker}_preds.csv")
            if os.path.exists(per_path):
                try:
                    pdf = pd.read_csv(per_path, index_col=0)
                    last = pdf.tail(1).to_dict(orient='records')[0]
                    text += "\nPrediction: " + json.dumps(last)
                except Exception:
                    pass
            elif os.path.exists(unified_preds):
                try:
                    up = pd.read_csv(unified_preds, index_col=0)
                    row = up[up.index.str.contains(ticker)].tail(1)
                    if not row.empty:
                        text += "\nPrediction: " + row.to_json(orient='records')
                except Exception:
                    pass

            texts.append(text)
            metadatas.append({"ticker": ticker})
        except Exception as e:
            print(f"Skipping {f}: {e}")

    if not texts:
        raise RuntimeError('No documents found to index')

    embeddings = OpenAIEmbeddings()
    faiss_index = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    faiss_index.save_local(index_dir)
    print(f"Saved FAISS index to {index_dir}")


if __name__ == '__main__':
    build_index_from_raw()
