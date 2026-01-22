"""
data_batch_downloader.py - Batch yfinance historical data downloader (multi-batch range)

Purpose:
    Downloads historical OHLCV data for one or multiple batch files in sequence.
    Supports range input: e.g., process batch_003.txt to batch_015.txt in one run.
    Automatically switches lookback period for intraday intervals (1m, 2m, 5m, etc.)
    to avoid Yahoo's 7-8 day limit.

New Usage Examples:
    python data_batch_downloader.py               # uses hardcoded START/END
    python data_batch_downloader.py 7             # only batch_007.txt
    python data_batch_downloader.py 1 20          # batches 001 to 020
    python data_batch_downloader.py 42 42         # just batch_042.txt

Configuration:
    Edit globals or use command-line args for flexibility.
    Sub-batches of 100 tickers per yf.download() call for reliability.

Last modified: January 2026
"""

from pathlib import Path
import time
import random
import sys
import pandas as pd
from tqdm import tqdm
import yfinance as yf

# Optional: better rate-limit resistance (pip install curl_cffi)
# from curl_cffi import requests 
# session = requests.Session(impersonate="chrome124")
# Then pass session=session to yf.download(...)

# ── GLOBAL CONFIGURATION ─────────────────────────────────────────────────────
INTERVAL = "1d"                         # "1d", "1wk", "1m", "5m", etc.

# Automatic lookback logic based on interval
if INTERVAL in ["1m", "2m", "5m", "15m", "30m", "60m", "90m"]:
    LOOKBACK_MODE = "recent"
    LOOKBACK_PERIOD = "7d"              # safe default: "7d", "30d", "60d"
    print(f"Using {LOOKBACK_MODE} mode for {INTERVAL} interval → period={LOOKBACK_PERIOD}")
else:
    LOOKBACK_MODE = "full"
    LOOKBACK_PERIOD = "2010-01-01"      # or "max", "10y", etc.
    print(f"Using {LOOKBACK_MODE} mode for {INTERVAL} interval → start={LOOKBACK_PERIOD}")

BASE_DIR = Path("data/yfinance/batches")
SAVE_DIRECTORY = Path("./data/yfinance") / INTERVAL
SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)

SUB_BATCH_SIZE = 100                    # Tickers per yf.download() call — safe value

# Default range if no command-line args provided
DEFAULT_BATCH_START = 1
DEFAULT_BATCH_END   = 17                 # Change these as fallback


def get_batch_range():
    """Determine which batches to process from command line or defaults"""
    if len(sys.argv) >= 3:
        try:
            start = int(sys.argv[1])
            end = int(sys.argv[2])
            if start < 1 or end < start:
                raise ValueError
            return start, end
        except:
            print("Invalid range. Usage: python script.py [start [end]]")
            sys.exit(1)
    elif len(sys.argv) == 2:
        try:
            num = int(sys.argv[1])
            return num, num
        except:
            print("Invalid batch number.")
            sys.exit(1)
    else:
        return DEFAULT_BATCH_START, DEFAULT_BATCH_END


def load_tickers(path: Path) -> list[str]:
    if not path.exists():
        print(f"Batch file not found: {path}")
        return []

    with open(path, encoding="utf-8") as f:
        tickers = [line.strip().upper() for line in f if line.strip() and not line.startswith('#')]

    if tickers:
        print(f"Loaded {len(tickers)} tickers from {path.name}")
    return tickers


def save_ticker_data(ticker: str, df: pd.DataFrame):
    """Simple per-ticker Parquet save."""
    if df.empty:
        return False
    file_path = SAVE_DIRECTORY / f"{ticker}.parquet"
    df.to_parquet(file_path, compression="snappy", index=True)
    return True


def process_single_batch(batch_num: int):
    batch_file = BASE_DIR / f"batch_{batch_num:03d}.txt"
    print(f"\n{'='*30} Processing batch {batch_num:03d} {'='*30}")
    print(f"File: {batch_file}")

    tickers = load_tickers(batch_file)
    if not tickers:
        return 0, len(tickers)  # success, failed

    # Optional skip list
    skip = {"AACB", "AACBU", "AAM", "ZIP", "ZOOZW"}
    tickers = [t for t in tickers if t not in skip]

    # Set fetch parameters based on mode
    if LOOKBACK_MODE == "recent":
        fetch_kwargs = {"period": LOOKBACK_PERIOD}
    else:
        fetch_kwargs = {"start": LOOKBACK_PERIOD}

    success = 0
    failed = 0

    for i in tqdm(range(0, len(tickers), SUB_BATCH_SIZE), desc=f"Batch {batch_num:03d} sub-batches", unit="sub"):
        sub_batch = tickers[i:i + SUB_BATCH_SIZE]

        try:
            data = yf.download(
                tickers=" ".join(sub_batch),
                interval=INTERVAL,
                **fetch_kwargs,
                group_by="ticker",
                threads=True,
                progress=False,
                auto_adjust=True,
                actions=False,
                ignore_tz=True,
                timeout=20,
                # session=session,  # uncomment if using curl_cffi
            )

            for ticker in sub_batch:
                try:
                    if ticker in data.columns.levels[0]:
                        df = data[ticker].dropna(how="all")
                        if not df.empty:
                            if save_ticker_data(ticker, df):
                                success += 1
                            else:
                                failed += 1
                        else:
                            failed += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"  {ticker:<10} save error: {str(e)}")
                    failed += 1

        except Exception as e:
            print(f"Sub-batch {i//SUB_BATCH_SIZE + 1} failed ({len(sub_batch)} tickers): {str(e)}")
            failed += len(sub_batch)

        time.sleep(1.5 + random.uniform(0, 2.5))  # polite delay

    print(f"Batch {batch_num:03d} done → Success: {success} | Failed/empty: {failed}")
    return success, failed


def main():
    SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    print(f"Save location: {SAVE_DIRECTORY.resolve()}\n")

    start_batch, end_batch = get_batch_range()
    print(f"Processing batches from {start_batch:03d} to {end_batch:03d}\n")

    total_success = 0
    total_failed = 0

    for batch_num in range(start_batch, end_batch + 1):
        succ, fail = process_single_batch(batch_num)
        total_success += succ
        total_failed += fail

        # Delay between different batch files
        if batch_num < end_batch:
            time.sleep(5 + random.uniform(0, 5))  # 5–10 sec between batch files

    print("\n" + "="*80)
    print("All requested batches finished")
    print(f"Total successfully saved : {total_success}")
    print(f"Total failed / empty     : {total_failed}")
    print("="*80)


if __name__ == "__main__":
    main()