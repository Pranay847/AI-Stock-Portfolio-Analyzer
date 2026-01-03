import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import time
import random

def fetch_stock_data_robust(tickers, period="5y", max_retries=3):
    """
    Fetch historical stock data with robust error handling
    
    Args:
        tickers: List of stock ticker symbols
        period: Time period (default: 5 years)
        max_retries: Maximum retry attempts per stock
    
    Returns:
        Dictionary with ticker as key and DataFrame as value
    """
    print(f"Fetching real-time data for {len(tickers)} stocks...")
    print(f"This will take approximately {len(tickers) * 3 / 60:.1f} minutes")
    stock_data = {}
    failed_stocks = []
    
    for i, ticker in enumerate(tickers):
        success = False
        retries = 0
        
        while not success and retries < max_retries:
            try:
                print(f"[{i+1}/{len(tickers)}] Downloading {ticker}... (attempt {retries+1})")
                
                # Create ticker object
                stock = yf.Ticker(ticker)
                
                # Download historical data
                df = stock.history(period=period)
                
                if not df.empty and len(df) > 100:  # Ensure we have enough data
                    stock_data[ticker] = df
                    print(f"✓ {ticker}: {len(df)} days of data")
                    success = True
                else:
                    print(f"⚠ {ticker}: Insufficient data (only {len(df)} rows)")
                    retries += 1
                    time.sleep(2)
                
            except Exception as e:
                print(f"✗ {ticker}: Error - {str(e)}")
                retries += 1
                
                if retries < max_retries:
                    wait_time = 5 * retries  # Exponential backoff
                    print(f"  Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        if not success:
            failed_stocks.append(ticker)
            print(f"✗✗ {ticker}: Failed after {max_retries} attempts")
        
        # Random delay between 2-4 seconds to avoid rate limiting
        if i < len(tickers) - 1:  # Don't wait after last stock
            delay = random.uniform(2, 4)
            time.sleep(delay)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Successful: {len(stock_data)}/{len(tickers)}")
    if failed_stocks:
        print(f"✗ Failed: {len(failed_stocks)} stocks")
        print(f"Failed tickers: {', '.join(failed_stocks)}")
    print(f"{'='*60}")
    
    return stock_data, failed_stocks

def save_stock_data(stock_data, output_dir="data/raw"):
    """Save stock data to CSV files"""
    os.makedirs(output_dir, exist_ok=True)
    
    for ticker, df in stock_data.items():
        filepath = os.path.join(output_dir, f"{ticker}.csv")
        df.to_csv(filepath)
        print(f"Saved {ticker} to {filepath}")

def download_in_batches(tickers, batch_size=10, period="5y"):
    """
    Download stocks in small batches with breaks
    
    This helps avoid rate limiting by taking breaks between batches
    """
    all_stock_data = {}
    all_failed = []
    
    # Split into batches
    batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
    
    print(f"Downloading {len(tickers)} stocks in {len(batches)} batches of {batch_size}")
    
    for batch_num, batch in enumerate(batches, 1):
        print(f"\n{'='*60}")
        print(f"BATCH {batch_num}/{len(batches)}")
        print(f"{'='*60}")
        
        stock_data, failed = fetch_stock_data_robust(batch, period=period)
        all_stock_data.update(stock_data)
        all_failed.extend(failed)
        
        # Take a break between batches (except after last batch)
        if batch_num < len(batches):
            break_time = 30
            print(f"\n⏸ Taking a {break_time} second break before next batch...")
            time.sleep(break_time)
    
    return all_stock_data, all_failed

if __name__ == "__main__":
    # List of stocks to download
    tickers = [
        # Tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
        "TSLA", "NVDA", "NFLX", "ADBE", "CRM",
        # Finance
        "JPM", "BAC", "WFC", "GS", "MS",
        # Healthcare
        "JNJ", "UNH", "PFE", "ABBV", "TMO",
        # Consumer
        "WMT", "PG", "KO", "COST", "MCD",
        # Others
        "DIS", "V", "MA", "INTC", "AMD"
    ]
    
    print("="*60)
    print("YAHOO FINANCE REAL-TIME DATA DOWNLOADER")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Download in batches
    stock_data, failed = download_in_batches(tickers, batch_size=5, period="5y")
    
    # Save successful downloads
    if stock_data:
        print(f"\nSaving {len(stock_data)} stocks to CSV...")
        save_stock_data(stock_data)
        print("\n✓ Data saved to data/raw/")
    
    # Save list of failed stocks for retry
    if failed:
        failed_file = "data/failed_tickers.txt"
        with open(failed_file, 'w') as f:
            f.write('\n'.join(failed))
        print(f"\n⚠ Failed tickers saved to {failed_file}")
        print("You can retry these later")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
