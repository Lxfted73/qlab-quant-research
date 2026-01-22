"""
Stock Timeframe Performance Analyzer – DuckDB + Local Parquet Only
──────────────────────────────────────────────────────────────────────────────

Analyzes pre-downloaded 1-minute OHLCV parquet files using DuckDB.
Shows % of up bars (Close > Open) across intraday timeframes.
Filters low-volume tickers and applies configurable trend filters
using LOCAL daily parquet files (no live API calls).

Last modified: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
from pandas_market_calendars import get_calendar
from tqdm import tqdm
import duckdb
import traceback

# ── CONFIGURATION ───────────────────────────────────────────────────────────

BASE_DATA_DIR    = Path("data/yfinance")
DB_PATH          = Path("qlab/data/qlab.duckdb")
DAILY_DATA_DIR   = BASE_DATA_DIR / "1d"

MIN_DAILY_VOLUME = 5_000_000
MIN_THRESHOLD    = 60
HIGH_THRESHOLD   = 80
MED_THRESHOLD    = 70

TARGET_DATE      = "2026-01-15"   # or None

INTERVALS = ['1min', '2min', '3min', '5min', '7min', '15min', '30min', '60min', '120min', '240min']
INTERVAL_LABELS = ['1min', '2min', '3min', '5min', '7min', '15min', '30min', '1h', '2h', '4h']

HEATMAP_SAVE_MODE    = "overall_only"
HEATMAP_OUTPUT_DIR   = Path("heatmaps")
HEATMAP_MAX_TICKERS  = 200
TOP_PER_GROUP        = 15

FILTERS = {
    "min_bars_for_sma": 50,
    "min_bars_for_rsi": 25,
    "use_sma_filter": False,
    "sma_period": 50,
    "must_be_above_sma": True,
    "use_rsi_filter": True,
    "rsi_period": 14,
    "rsi_threshold": 50,
    "rsi_direction": "above",
}

plt.rcParams['figure.max_open_warning'] = 0

# ── Daily Loader & Filters (unchanged from your working version) ────────────
def load_daily_data(ticker: str) -> pd.DataFrame | None:
    path = DAILY_DATA_DIR / f"{ticker.upper()}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            if "Datetime" in df.columns:
                df = df.set_index("Datetime")
            elif "Date" in df.columns:
                df = df.set_index("Date")
            else:
                return None
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.sort_index()
        if "Close" not in df.columns:
            return None
        return df[["Close"]].copy()
    except:
        return None

def is_above_sma(ticker: str) -> bool:
    if not FILTERS["use_sma_filter"]: return True
    df = load_daily_data(ticker)
    if df is None or len(df) < FILTERS["min_bars_for_sma"]: return False
    closes = df["Close"].tail(FILTERS["sma_period"])
    if len(closes) < FILTERS["sma_period"]: return False
    sma = closes.mean()
    latest = closes.iloc[-1]
    return latest > sma if FILTERS["must_be_above_sma"] else latest < sma

def get_rsi(ticker: str) -> float | None:
    if not FILTERS["use_rsi_filter"]: return 50.0
    df = load_daily_data(ticker)
    if df is None or len(df) < FILTERS["min_bars_for_rsi"]: return None
    closes = df["Close"].values
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    period = FILTERS["rsi_period"]
    if len(gains) < period: return None
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
    return rsi

def passes_filters(ticker: str) -> bool:
    if not FILTERS["use_sma_filter"] and not FILTERS["use_rsi_filter"]:
        return True
    sma_ok = is_above_sma(ticker)
    rsi = get_rsi(ticker)
    if rsi is None:
        rsi_ok = False
    else:
        rsi_ok = (rsi > FILTERS["rsi_threshold"]) if FILTERS["rsi_direction"] == "above" else (rsi < FILTERS["rsi_threshold"])
    return sma_ok and rsi_ok

# ── DuckDB Setup ────────────────────────────────────────────────────────────
def setup_qlab_duckdb(db_path: Path = DB_PATH, read_only=False):
    con = duckdb.connect(str(db_path), read_only=read_only)
    print(f"Connected to DuckDB: {db_path.resolve()}")

    con.execute("SET threads TO 8;")

    con.execute("DROP VIEW IF EXISTS bars_1min;")

    parquet_folder = BASE_DATA_DIR / "1m"
    if not parquet_folder.exists() or not any(parquet_folder.glob("*.parquet")):
        raise ValueError(f"No 1m files in {parquet_folder}")

    query = f"""
    CREATE OR REPLACE VIEW bars_1min AS
    SELECT
        '1m' AS interval,
        regexp_replace(
            regexp_replace(filename, '^.*[\\\\/]', ''),     -- remove path
            '\\.parquet$', ''
        ) AS symbol,
        "Datetime"::TIMESTAMP AS ts,
        "Open"::DOUBLE AS open,
        High::DOUBLE AS high,
        Low::DOUBLE AS low,
        "Close"::DOUBLE AS close,
        Volume::BIGINT AS volume,
        filename AS source_file
    FROM read_parquet(
        '{parquet_folder}/*.parquet',
        filename = true,
        union_by_name = true
    )
    WHERE "Datetime" IS NOT NULL
    """

    con.execute(query)
    print("✓ View created with clean symbol extraction")

    # Debug: check symbols
    print("\nFirst 10 unique symbols in view (should be clean):")
    print(con.sql("SELECT DISTINCT symbol FROM bars_1min LIMIT 10").df())

    return con

# ── Helpers ─────────────────────────────────────────────────────────────────
def get_trading_day(target_date_str: str | None = None):
    nyse = get_calendar('NYSE')
    today = pd.Timestamp.now(tz='America/New_York').normalize()

    if target_date_str:
        try:
            target = pd.Timestamp(target_date_str, tz='America/New_York').normalize()
            schedule = nyse.schedule(start_date=target - timedelta(days=10), end_date=target + timedelta(days=1))
            if target.date() in schedule.index.date:
                print(f"Using specified date: {target.date()}")
                return target.date()
        except:
            print(f"Invalid date '{target_date_str}'. Falling back...")

    schedule = nyse.schedule(start_date=today - timedelta(days=40), end_date=today)
    if len(schedule) == 0:
        raise ValueError("No trading days found")

    prev = schedule.index[-1].date() if today.date() not in schedule.index.date else schedule.index[-2].date()
    print(f"Using last trading day: {prev}")
    return prev

def get_interval_minutes(interval: str) -> int:
    if 'min' in interval: return int(interval.replace('min', ''))
    if 'h' in interval: return int(interval.replace('h', '')) * 60
    raise ValueError(f"Bad interval: {interval}")

# ── Main ────────────────────────────────────────────────────────────────────
def main():
    con = None
    try:
        con = setup_qlab_duckdb()

        date = get_trading_day(TARGET_DATE)
        date_str = date.strftime('%Y-%m-%d')

        market_open  = f"{date_str} 09:30:00"
        market_close = f"{date_str} 16:00:00"

        output_subdir = HEATMAP_OUTPUT_DIR / date_str
        output_subdir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Analyzing {date_str} ===")
        print(f"  Volume ≥ {MIN_DAILY_VOLUME:,}")
        print(f"  Up-bar ≥ {MIN_THRESHOLD}%")

        # Debug: check data availability
        print("\nData availability check:")
        print(con.sql(f"""
            SELECT 
                COUNT(*) AS rows_in_day,
                COUNT(DISTINCT symbol) AS symbols,
                COUNT(DISTINCT ts) AS unique_times,
                MIN(ts) AS first,
                MAX(ts) AS last
            FROM bars_1min
            WHERE ts::DATE = '{date_str}'
        """).df())

        # High-volume symbols
        print("\nFinding high-volume symbols...")
        high_vol_df = con.sql(f"""
            SELECT symbol, SUM(volume) AS total_vol
            FROM bars_1min
            WHERE ts::DATE = '{date_str}'
              AND ts BETWEEN '{market_open}' AND '{market_close}'
            GROUP BY symbol
            HAVING total_vol >= {MIN_DAILY_VOLUME}
            ORDER BY total_vol DESC
        """).df()

        candidates = high_vol_df['symbol'].tolist()
        print(f"→ {len(candidates):,} candidates")

        if not candidates:
            print("No high-volume symbols found.")
            return

        # Apply filters
        print(f"\nApplying filters ({len(candidates)} candidates)...")
        filtered = [s for s in tqdm(candidates, desc="Filtering", unit="ticker", ncols=100)
                    if passes_filters(s)]

        print(f"\n→ {len(filtered)} passed filters")
        if filtered:
            print("Sample:", ", ".join(filtered[:10]))
        else:
            print("No symbols passed filters.")
            return

        # Resampling
        print("\nResampling...")
        results = []

        for iv in tqdm(INTERVALS, desc="Timeframes", unit="tf", ncols=100):
            mins = get_interval_minutes(iv)

            # Safe bucketing using modulo on minutes since midnight
            bucket_expr = f"""
                date_trunc('day', ts) + 
                INTERVAL '{mins} minutes' * FLOOR(
                    (EXTRACT(EPOCH FROM (ts - date_trunc('day', ts))) / 60) / {mins}
                )
            """ if mins > 1 else "date_trunc('minute', ts)"

            q = f"""
                WITH bars AS (
                    SELECT symbol, ts, open, close
                    FROM bars_1min
                    WHERE symbol IN ({','.join(f"'{s}'" for s in filtered)})
                      AND ts::DATE = '{date_str}'
                      AND ts BETWEEN '{market_open}' AND '{market_close}'
                ),
                resampled AS (
                    SELECT 
                        symbol,
                        {bucket_expr} AS bucket,
                        FIRST(open) AS open,
                        LAST(close) AS close
                    FROM bars
                    GROUP BY symbol, bucket
                )
                SELECT 
                    symbol,
                    COUNT(*) AS n_bars,
                    SUM(CASE WHEN close > open THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS up_pct
                FROM resampled
                GROUP BY symbol
                HAVING n_bars >= 3
            """

            try:
                df_iv = con.sql(q).df()
                if df_iv.empty:
                    print(f"  {iv:>6} → 0 bars")
                else:
                    print(f"  {iv:>6} → {len(df_iv)} symbols")
                    df_iv['interval'] = iv
                    results.append(df_iv)
            except Exception as e:
                print(f"  {iv:>6} failed: {e}")
                continue

        if not results:
            print("No data after resampling.")
            return

        # Combine & stats
        df = pd.concat(results).pivot(
            index='symbol', columns='interval', values='up_pct'
        ).reindex(columns=INTERVALS).rename(columns=dict(zip(INTERVALS, INTERVAL_LABELS)))

        df['Max_%'] = df.max(axis=1)
        df['Best_TF'] = df.idxmax(axis=1)
        df['Strong_cnt'] = (df[INTERVAL_LABELS] >= HIGH_THRESHOLD).sum(axis=1)
        df['Good_cnt'] = (df[INTERVAL_LABELS] >= MED_THRESHOLD).sum(axis=1)

        df = df.sort_values('Max_%', ascending=False)

        print(f"\nQualified: {len(df)} symbols")

        # Heatmaps (your original logic)
        df_show = df[df['Max_%'] >= MIN_THRESHOLD]

        if HEATMAP_SAVE_MODE in ["all", "groups_only"]:
            for tf, group in df_show.groupby('Best_TF'):
                if len(group) == 0: continue
                top = group.sort_values('Max_%', ascending=False).head(TOP_PER_GROUP)
                print(f"\nBest {tf}: {len(group)} (top {TOP_PER_GROUP})")
                print(top[['Max_%'] + INTERVAL_LABELS].round(1))

                fig_h = max(5, min(18, len(top) * 0.45))
                plt.figure(figsize=(14, fig_h))
                sns.heatmap(top[INTERVAL_LABELS], annot=True, fmt=".1f", cmap="YlGnBu",
                            vmin=50, vmax=100, linewidths=0.4, annot_kws={"size":10})
                plt.title(f"{tf} – {date_str}")
                plt.ylabel("Ticker"); plt.xlabel("Timeframe")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                safe_tf = tf.replace(" ", "").replace(":", "")
                path = output_subdir / f"heatmap_{date_str}_{safe_tf}_top{len(top)}.png"
                plt.savefig(path, dpi=140, bbox_inches='tight')
                plt.close()
                print(f"  → {path}")

        if HEATMAP_SAVE_MODE in ["all", "overall_only"]:
            data = df_show[INTERVAL_LABELS]
            if len(data) > HEATMAP_MAX_TICKERS:
                data = data.loc[data.max(axis=1).nlargest(HEATMAP_MAX_TICKERS).index]

            if not data.empty:
                fig_h = max(6, min(48, len(data) * 0.38))
                plt.figure(figsize=(18, fig_h))
                sns.heatmap(data, annot=True, fmt=".1f", cmap="YlGnBu",
                            vmin=50, vmax=100, linewidths=0.3, annot_kws={"size":8})
                plt.title(f"Overall – {date_str} (top {len(data)})")
                plt.ylabel("Ticker"); plt.xlabel("Timeframe")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                path = HEATMAP_OUTPUT_DIR / f"heatmap_{date_str}_overall_top{len(data)}.png"
                plt.savefig(path, dpi=200, bbox_inches='tight')
                plt.close()
                print(f"\n✓ Overall: {path}")

        print("\nDone!")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        if con is not None:
            con.close()
            print("DuckDB closed.")

if __name__ == "__main__":
    main()