#!/usr/bin/env python3
# quality_scan_and_cleanup.py
"""
Quality Scan + Aggressive Cleanup — Any Timeframe Parquet Folder
───────────────────────────────────────────────────────────────

1. Scans all .parquet files in data/yfinance/<interval>/
2. Generates fresh quality report
3. Immediately runs aggressive cleanup using that report:
   - ≤5 rows → junk/very_short
   - negative/zero Close → junk/bad_prices
   - high zero-volume suspicious → junk/high_zero_vol_aggressive
     (zero_vol_days > 60 AND (rows < 1500 OR zero_vol_pct > 0.20))

Usage:
    python quality_scan_and_cleanup.py 1d
    python quality_scan_and_cleanup.py 1wk
    python quality_scan_and_cleanup.py 1m

After running:
- New report saved as <interval>_quality_report.csv
- Junk moved to data/yfinance/junk_removed/<interval>/...
- Ready for next step (universe filtering → backtest)

Last modified: February 2026
"""

import sys
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import shutil
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_INTERVAL = "1d"

MIN_ROWS_FLAG_SHORT     = 5
ZERO_VOL_THRESHOLD_DAYS = 60
ZERO_VOL_PCT_THRESHOLD  = 0.20
ROWS_AGGRESSIVE_FILTER  = 1500

# ── Path / Interval helpers ──────────────────────────────────────────────────

def get_interval():
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip().lower().replace(" ", "")
        valid = {"1m","2m","3m","5m","15m","30m","60m","90m","1h","1d","1wk","1mo"}
        if arg in valid:
            return arg
    return DEFAULT_INTERVAL


def get_paths(interval: str):
    folder_name = interval.replace(" ", "").lower()
    data_dir    = Path("data/yfinance") / folder_name
    report_path = Path(f"{folder_name}_quality_report.csv")

    junk_root = Path("data/yfinance/junk_removed") / folder_name
    junk_short    = junk_root / "very_short"
    junk_bad      = junk_root / "bad_prices"
    junk_illiquid = junk_root / "high_zero_vol_aggressive"

    for d in [junk_root, junk_short, junk_bad, junk_illiquid]:
        d.mkdir(parents=True, exist_ok=True)

    return data_dir, report_path, junk_short, junk_bad, junk_illiquid


def move_file(ticker: str, reason: str, data_dir: Path, junk_short, junk_bad, junk_illiquid):
    src = data_dir / f"{ticker}.parquet"
    if not src.exists():
        return False, "file not found"

    if "short" in reason:
        dst_folder = junk_short
    elif "bad" in reason:
        dst_folder = junk_bad
    else:
        dst_folder = junk_illiquid

    dst = dst_folder / src.name
    try:
        shutil.move(str(src), str(dst))
        return True, dst.parent.name
    except Exception as e:
        return False, str(e)


# ── Quality scan ──────────────────────────────────────────────────────────────

def scan_quality(data_dir: Path):
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        print(f"No .parquet files found in {data_dir}")
        return None

    print(f"Scanning {len(files):,} files ...\n")

    results = []

    for p in tqdm(files, desc="Scanning"):
        ticker = p.stem.upper()
        try:
            df = pd.read_parquet(p)
            n = len(df)
            if n == 0:
                results.append({"ticker": ticker, "rows": 0, "flag": "empty"})
                continue

            idx = df.index
            start = idx.min()
            end   = idx.max()
            span  = (end - start).days if n > 1 else 0

            closes  = df.get("Close", pd.Series(dtype=float))
            volumes = df.get("Volume", pd.Series(dtype=float))

            row = {
                "ticker": ticker,
                "rows": n,
                "start_date": start.date() if pd.notnull(start) else None,
                "end_date":   end.date()   if pd.notnull(end)   else None,
                "span_days": span,
                "nan_close": closes.isna().sum(),
                "bad_close": (closes <= 0).sum(),
                "zero_vol_days": (volumes <= 0).sum(),
                "dup_index": idx.duplicated().sum(),
            }

            if row["rows"] <= MIN_ROWS_FLAG_SHORT:
                row["flag"] = "very_short"
            elif row["bad_close"] > 0:
                row["flag"] = "bad_prices"
            elif row["zero_vol_days"] > ZERO_VOL_THRESHOLD_DAYS:
                row["flag"] = "high_zero_vol"
            else:
                row["flag"] = "ok"

            results.append(row)

        except Exception as e:
            results.append({"ticker": ticker, "flag": f"read_error: {str(e)}"})

    df = pd.DataFrame(results)
    df["zero_vol_pct"] = df["zero_vol_days"] / df["rows"].replace(0, 1)

    print("\nSummary by flag:")
    print(df["flag"].value_counts(dropna=False))

    return df


# ── Cleanup ───────────────────────────────────────────────────────────────────

def perform_cleanup(df: pd.DataFrame, data_dir: Path, junk_short, junk_bad, junk_illiquid):
    moved = {"very_short": 0, "bad_prices": 0, "high_zero_vol_aggressive": 0}

    # Very short
    short = df[df["flag"] == "very_short"]
    print(f"\nMoving very short (≤{MIN_ROWS_FLAG_SHORT} rows): {len(short)}")
    for _, r in short.iterrows():
        ok, msg = move_file(r["ticker"], "very_short", data_dir, junk_short, junk_bad, junk_illiquid)
        if ok:
            moved["very_short"] += 1
            print(f"  {r['ticker']:<8} → {msg}")

    # Bad prices
    bad = df[df["bad_close"] > 0]
    print(f"\nMoving bad prices: {len(bad)}")
    for _, r in bad.iterrows():
        ok, msg = move_file(r["ticker"], "bad_prices", data_dir, junk_short, junk_bad, junk_illiquid)
        if ok:
            moved["bad_prices"] += 1
            print(f"  {r['ticker']:<8} → {msg}")

    # Aggressive illiquid
    illiq = df[
        (df["flag"] == "high_zero_vol") &
        (df["zero_vol_days"] > ZERO_VOL_THRESHOLD_DAYS) &
        ((df["rows"] < ROWS_AGGRESSIVE_FILTER) | (df["zero_vol_pct"] > ZERO_VOL_PCT_THRESHOLD))
    ]
    print(f"\nMoving aggressive illiquid: {len(illiq)}")
    for _, r in illiq.iterrows():
        ok, msg = move_file(r["ticker"], "high_zero_vol_aggressive", data_dir, junk_short, junk_bad, junk_illiquid)
        if ok:
            moved["high_zero_vol_aggressive"] += 1
            print(f"  {r['ticker']:<8} → {msg}")

    return moved


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    interval = get_interval()
    print(f"QLab Quality Scan + Cleanup • {interval.upper()} • {datetime.now():%Y-%m-%d %H:%M}\n")

    data_dir, report_path, js, jb, ji = get_paths(interval)

    # 1. Scan + save report
    df = scan_quality(data_dir)
    if df is None:
        return

    df.to_csv(report_path, index=False)
    print(f"\nReport saved → {report_path}")

    # 2. Cleanup using fresh df
    moved = perform_cleanup(df, data_dir, js, jb, ji)

    # Summary
    total_moved = sum(moved.values())
    print("\n" + "═" * 80)
    print(f"Done. Moved {total_moved} files:")
    for cat, cnt in moved.items():
        if cnt > 0:
            print(f"  • {cat:28} : {cnt:4d}")
    print(f"\nClean folder: {data_dir}")
    print(f"Junk:         {js.parent}")
    print("Next:")
    print("  • Review new report")
    print("  • Build filtered universe list")
    print("  • Start factor backtest")
    print("═" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"\nError: {type(e).__name__}: {e}")