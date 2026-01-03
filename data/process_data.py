import os
import glob
import json
import pandas as pd
from data.features import compute_technical_indicators, add_rolling_stats


def process_all_raw(raw_dir='data/raw', processed_dir='data/processed'):
    os.makedirs(processed_dir, exist_ok=True)
    files = glob.glob(os.path.join(raw_dir, '*.csv'))
    for f in files:
        try:
            ticker = os.path.splitext(os.path.basename(f))[0]
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            df = compute_technical_indicators(df)
            df = add_rolling_stats(df)

            # attach metadata if exists
            meta_path = os.path.join(raw_dir, f"{ticker}_meta.json")
            meta = {}
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as fh:
                    meta = json.load(fh)
            for k, v in meta.items():
                df[k] = v

            out_path = os.path.join(processed_dir, f"{ticker}.csv")
            df.to_csv(out_path)
            print(f"Processed {ticker} -> {out_path}")
        except Exception as e:
            print(f"Skipping {f}: {e}")


if __name__ == '__main__':
    process_all_raw()
