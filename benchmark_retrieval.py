"""
Real before/after benchmark for the data-retrieval optimizations.

Run this on a machine WITH network access to Yahoo Finance and HuggingFace
(i.e. not inside a locked-down sandbox). It measures the three things the
optimization changed, each isolated so the numbers are attributable:

  1. FETCH      sequential per-stock   vs  concurrent (ThreadPoolExecutor)
  2. EMBED      per-item encode()      vs  one batched encode()
  3. DB WRITE   N upserts              vs  1 batched upsert

Every section is a REAL measurement — no stubs, no assumed latencies.
Sections degrade gracefully: if the network or model is unavailable, that
section is skipped with a clear message instead of failing the whole run.

Usage:
    python benchmark_retrieval.py                 # default 10-stock portfolio
    python benchmark_retrieval.py AAPL MSFT NVDA  # custom tickers
    BENCH_WORKERS=8 python benchmark_retrieval.py # override worker count
"""
import os
import sys
import time
import json
import shutil
import tempfile
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
                   "TSLA", "NVDA", "NFLX", "JPM", "V"]
WORKERS = int(os.getenv("BENCH_WORKERS", "10"))
EMBED_MODEL = "all-MiniLM-L6-v2"


def hr(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ----------------------------------------------------------------------
# Shared: fetch a single stock's `info` dict (same call the app makes)
# ----------------------------------------------------------------------
def fetch_one(symbol):
    import yfinance as yf
    info = yf.Ticker(symbol).info
    price = info.get("currentPrice", info.get("regularMarketPrice", 0))
    return {
        "symbol": symbol,
        "name": info.get("longName", symbol),
        "price": price,
        "previous_close": info.get("previousClose", 0),
        "market_cap": info.get("marketCap", 0),
        "pe_ratio": info.get("trailingPE", 0),
        "volume": info.get("volume", 0),
        "sector": info.get("sector", "Unknown"),
        "industry": info.get("industry", "Unknown"),
        "description": info.get("longBusinessSummary", ""),
        "timestamp": datetime.now().isoformat(),
    }


def to_text(d):
    return (f"{d['name']} ({d['symbol']}) Sector: {d['sector']} "
            f"Industry: {d['industry']} Price: ${d['price']} "
            f"Market Cap: ${d['market_cap']} P/E: {d['pe_ratio']} "
            f"{d['description'][:300]}")


# ----------------------------------------------------------------------
# PART 1 — FETCH: sequential vs concurrent (REAL network)
# ----------------------------------------------------------------------
def bench_fetch(tickers):
    hr("PART 1: FETCH  sequential vs concurrent  (real Yahoo Finance)")
    try:
        import yfinance  # noqa: F401
    except ImportError:
        print("  SKIPPED — yfinance not installed (`pip install yfinance`)")
        return None

    # sanity probe
    try:
        _ = fetch_one(tickers[0])
    except Exception as e:
        print(f"  SKIPPED — network/fetch failed: {type(e).__name__}: {str(e)[:120]}")
        return None

    # sequential
    t0 = time.perf_counter()
    seq_data = [fetch_one(s) for s in tickers]
    seq = time.perf_counter() - t0

    # concurrent
    t0 = time.perf_counter()
    con_data = [None] * len(tickers)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        fut = {ex.submit(fetch_one, s): i for i, s in enumerate(tickers)}
        for f in as_completed(fut):
            con_data[fut[f]] = f.result()
    con = time.perf_counter() - t0

    print(f"  tickers={len(tickers)}  workers={WORKERS}")
    print(f"  sequential : {seq:7.2f}s  ({seq / len(tickers):.2f}s/stock)")
    print(f"  concurrent : {con:7.2f}s  ({con / len(tickers):.2f}s/stock)")
    print(f"  speedup    : {seq / con:.1f}x")
    return con_data


# ----------------------------------------------------------------------
# PART 2 — EMBED: per-item vs batched encode (REAL model)
# ----------------------------------------------------------------------
def bench_embed(stock_data):
    hr("PART 2: EMBED  per-item vs batched encode  (real all-MiniLM-L6-v2)")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  SKIPPED — sentence-transformers not installed")
        return
    if not stock_data:
        print("  SKIPPED — no stock data from PART 1")
        return

    try:
        model = SentenceTransformer(EMBED_MODEL)
    except Exception as e:
        print(f"  SKIPPED — model load failed (HF blocked?): {type(e).__name__}: {str(e)[:100]}")
        return

    texts = [to_text(d) for d in stock_data]
    model.encode(["warmup"])  # exclude first-call / lazy-init overhead

    # per-item
    t0 = time.perf_counter()
    _ = [model.encode(t).tolist() for t in texts]
    per_item = time.perf_counter() - t0

    # batched
    t0 = time.perf_counter()
    _ = [v.tolist() for v in model.encode(texts, batch_size=32, show_progress_bar=False)]
    batched = time.perf_counter() - t0

    print(f"  texts={len(texts)}")
    print(f"  per-item encode : {per_item * 1000:7.1f} ms")
    print(f"  batched encode  : {batched * 1000:7.1f} ms")
    print(f"  speedup         : {per_item / batched:.1f}x")


# ----------------------------------------------------------------------
# PART 3 — DB WRITE: N upserts vs 1 batched upsert (REAL, local)
# ----------------------------------------------------------------------
def bench_dbwrite(n_items):
    hr("PART 3: DB WRITE  N upserts vs 1 batched upsert  (real ChromaDB)")
    try:
        import chromadb
    except ImportError:
        print("  SKIPPED — chromadb not installed")
        return
    import random

    def vec():
        return [random.random() for _ in range(384)]

    ids = [f"doc_{i}" for i in range(n_items)]
    embs = [vec() for _ in range(n_items)]
    docs = [f"document {i}" for i in range(n_items)]
    metas = [{"idx": i} for i in range(n_items)]

    base = tempfile.mkdtemp(prefix="chroma_bench_")
    try:
        c_old = chromadb.PersistentClient(path=os.path.join(base, "old")).get_or_create_collection("bench_coll")
        t0 = time.perf_counter()
        for i in range(n_items):
            c_old.upsert(ids=[ids[i]], embeddings=[embs[i]], documents=[docs[i]], metadatas=[metas[i]])
        old = time.perf_counter() - t0

        c_new = chromadb.PersistentClient(path=os.path.join(base, "new")).get_or_create_collection("bench_coll")
        t0 = time.perf_counter()
        c_new.upsert(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
        new = time.perf_counter() - t0
    finally:
        shutil.rmtree(base, ignore_errors=True)

    print(f"  items={n_items}")
    print(f"  {n_items} upserts    : {old * 1000:7.1f} ms")
    print(f"  1 batched upsert : {new * 1000:7.1f} ms")
    print(f"  speedup          : {old / new:.1f}x")


def main():
    tickers = [t.upper() for t in sys.argv[1:]] or DEFAULT_TICKERS
    hr(f"RETRIEVAL BENCHMARK  ({len(tickers)} tickers, {WORKERS} workers)")
    print("  All sections are real measurements; unavailable ones are skipped.")

    stock_data = bench_fetch(tickers)
    bench_embed(stock_data)
    bench_dbwrite(len(tickers))

    hr("DONE")
    print("  Paste the three speedup lines above as your before/after numbers.")


if __name__ == "__main__":
    main()
