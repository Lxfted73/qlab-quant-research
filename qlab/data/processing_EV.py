"""
Stock Timeframe Performance Analyzer – Single File Version with Progress Bar
──────────────────────────────────────────────────────────────────────────────

Analyzes pre-downloaded 1-minute OHLCV parquet files.
Shows % of up bars (Close > Open) across intraday timeframes for a specified or previous trading day.
Shows only top 15 stocks per "best timeframe" group + separate heatmaps per group.
Filters out low-volume tickers (<5M shares daily) for scalping suitability.
Saves heatmaps: group ones in day subfolder, overall one directly in HEATMAP_OUTPUT_DIR.

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

# ── Configuration ───────────────────────────────────────────────────────────
DATA_DIR             = Path("data/yfinance/1m")
MIN_THRESHOLD        = 60                                # % up bars threshold to include
HIGH_THRESHOLD       = 80                                # "strong"
MED_THRESHOLD        = 70                                # "good"
MAX_SYMBOLS          = 7000                              # safety limit
SAVE_INTERMEDIATE_CSV = True
CSV_OUTPUT_SUFFIX    = "strong_stocks"
HEATMAP_MAX_TICKERS  = 120                               # max rows in overall heatmap
TOP_PER_GROUP        = 15                                # max stocks shown per group

# Volume filter for scalping (adjust as needed)
MIN_DAILY_VOLUME     = 5_000_000                         # 5M shares — good for 5–10 trades/hour

# Custom date override (set to None for previous trading day)
# Format: "YYYY-MM-DD" string, e.g. "2026-01-15"
TARGET_DATE          = "2026-01-12"

# Controls which heatmaps to save
HEATMAP_SAVE_MODE    = "overall_only"                    # Options: "all", "overall_only", "groups_only", "none"

# Base folder for heatmaps
HEATMAP_OUTPUT_DIR   = Path("heatmaps")                  # ← change this to your preferred base folder

INTERVALS = ['1min', '2min', '3min', '5min', '7min', '15min', '30min', '60min', '120min', '240min']
INTERVAL_LABELS = ['1min', '2min', '3min', '5min', '7min', '15min', '30min', '1h', '2h', '4h']

MARKET_OPEN  = '09:30:00'
MARKET_CLOSE = '16:00:00'

plt.rcParams['figure.max_open_warning'] = 0


# ── Helpers ─────────────────────────────────────────────────────────────────
def get_trading_day(target_date_str: str | None = None):
    nyse = get_calendar('NYSE')
    today = pd.Timestamp.now(tz='America/New_York').normalize()

    if target_date_str:
        try:
            target = pd.Timestamp(target_date_str, tz='America/New_York').normalize()
            schedule = nyse.schedule(start_date=target - timedelta(days=10), end_date=target + timedelta(days=1))
            if target.date() in schedule.index.date:
                print(f"Using specified trading day: {target.date()}")
                return target.date()
            else:
                print(f"Warning: {target_date_str} is not a trading day. Falling back to previous valid day.")
        except Exception as e:
            print(f"Invalid date format '{target_date_str}': {e}. Falling back to previous trading day.")

    # Fallback: previous trading day
    schedule = nyse.schedule(start_date=today - timedelta(days=40), end_date=today)
    if len(schedule) < 2:
        return None
    return schedule.index[-2].date()


def load_ticker_1min(ticker: str) -> pd.DataFrame | None:
    path = DATA_DIR / f"{ticker.upper()}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'index' in df.columns:
                df = df.set_index('index')
            else:
                df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
        return df
    except Exception as e:
        print(f"Error loading {ticker}: {e}")
        return None


def analyze_day(df_1min: pd.DataFrame, target_date: datetime.date) -> dict | None:
    if df_1min is None or df_1min.empty:
        return None

    df_et = df_1min.tz_convert('America/New_York')
    start = pd.Timestamp(f"{target_date} {MARKET_OPEN}", tz='America/New_York')
    end   = pd.Timestamp(f"{target_date} {MARKET_CLOSE}", tz='America/New_York')

    df_day = df_et.loc[start:end]
    if df_day.empty:
        return None

    # ── Volume filter for scalping suitability ──────────────────────────────
    daily_volume = df_day['Volume'].sum()
    if daily_volume < MIN_DAILY_VOLUME:
        return None

    results = {}
    for intv, label in zip(INTERVALS, INTERVAL_LABELS):
        resampled = df_day.resample(intv).agg({
            'Open':   'first',
            'High':   'max',
            'Low':    'min',
            'Close':  'last',
            'Volume': 'sum'
        }).dropna(how='all')

        if resampled.empty:
            results[label] = 0.0
            continue

        up_count = (resampled['Close'] > resampled['Open']).sum()
        total = len(resampled)
        pct_up = (up_count / total) * 100 if total > 0 else 0.0
        results[label] = round(pct_up, 1)

    return results


def main():
    target_date = get_trading_day(TARGET_DATE)
    if not target_date:
        print("Could not determine a valid trading day.")
        return

    print(f"\nAnalyzing trading day: {target_date}\n")

    parquet_files = list(DATA_DIR.glob("*.parquet"))
    all_symbols = [f.stem.upper() for f in parquet_files][:MAX_SYMBOLS]

    print(f"Found {len(parquet_files)} files → analyzing up to {len(all_symbols)} tickers\n")

    all_results = {}
    skipped = []

    start_time = datetime.now()
    with tqdm(total=len(all_symbols), desc="Processing tickers", unit="ticker",
              dynamic_ncols=True, mininterval=1.0) as pbar:

        for symbol in all_symbols:
            df = load_ticker_1min(symbol)
            if df is None:
                skipped.append(symbol)
                pbar.update(1)
                continue

            day_stats = analyze_day(df, target_date)
            if day_stats is None:
                skipped.append(symbol)
                pbar.update(1)
                continue

            all_results[symbol] = day_stats

            elapsed = datetime.now() - start_time
            processed = pbar.n + 1
            rate = processed / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0.0
            remaining = pbar.total - processed
            eta_min = (remaining / rate / 60) if rate > 0 else "--"
            pbar.set_postfix_str(
                f"{symbol:8} | {rate:5.1f} t/s | ETA ~{eta_min} min",
                refresh=True
            )
            pbar.update(1)

    print(f"\nCompleted: {len(all_results)} analyzed, {len(skipped)} skipped\n")

    if skipped:
        print(f"Skipped (first 60): {', '.join(skipped[:60])}{'...' if len(skipped)>60 else ''}\n")

    if not all_results:
        print("No usable data found.")
        return

    # ── Build results DataFrame ─────────────────────────────────────────────
    df_results = pd.DataFrame.from_dict(all_results, orient='index')
    df_results = df_results[INTERVAL_LABELS]

    # Filter: ≥ MIN_THRESHOLD on at least one timeframe
    good_mask = df_results.max(axis=1) >= MIN_THRESHOLD
    df_filtered = df_results[good_mask].copy()

    if df_filtered.empty:
        print(f"No stocks reached ≥{MIN_THRESHOLD}% up bars on {target_date}.")
        return

    print(f"Found {len(df_filtered)} stocks with ≥{MIN_THRESHOLD}% up bars somewhere\n")

    # Add summary columns
    df_filtered['Max_%'] = df_filtered.max(axis=1).round(1)
    df_filtered['Strong_count'] = (df_filtered >= HIGH_THRESHOLD).sum(axis=1)
    df_filtered['Good_count']   = (df_filtered >= MED_THRESHOLD).sum(axis=1)

    # Sort strongest first (global sort)
    df_filtered = df_filtered.sort_values('Max_%', ascending=False)

    # ── Save full filtered results early ────────────────────────────────────
    if SAVE_INTERMEDIATE_CSV:
        csv_path = f"{CSV_OUTPUT_SUFFIX}_{target_date}.csv"
        save_df = df_filtered.copy()
        save_df.insert(0, 'Ticker', save_df.index)
        save_df.to_csv(csv_path, index=False)
        print(f"Saved full filtered results → {csv_path} ({len(save_df)} rows)\n")

    # ── Prepare date-specific output subfolder (only for group heatmaps) ─────
    date_str = str(target_date)
    output_subdir = HEATMAP_OUTPUT_DIR / date_str
    output_subdir.mkdir(parents=True, exist_ok=True)

    # ── Terminal print with color (full table) ──────────────────────────────
    print("═"*100)
    print(f" Stocks with ≥{MIN_THRESHOLD}% up bars on at least one timeframe ".center(100, "═"))
    print("═"*100)

    def format_cell(x):
        x = round(x, 1)
        if x >= HIGH_THRESHOLD:
            return f"\033[92m{x:.1f}\033[0m"   # green
        elif x >= MED_THRESHOLD:
            return f"\033[93m{x:.1f}\033[0m"   # yellow
        else:
            return f"{x:.1f}"

    print_df = df_filtered.drop(columns=['Max_%', 'Strong_count', 'Good_count']).copy()
    for col in print_df.columns:
        print_df[col] = print_df[col].apply(format_cell)

    print_df['Max_%'] = df_filtered['Max_%'].apply(lambda x: f"\033[1m{x:.1f}\033[0m")
    print_df['Strong'] = df_filtered['Strong_count'].apply(lambda x: f"\033[92m{x}\033[0m" if x > 0 else x)
    print_df['Good']   = df_filtered['Good_count'].apply(lambda x: f"\033[93m{x}\033[0m" if x > 0 else x)

    print(print_df.to_string())
    print("\nLegend: \033[92mgreen ≥80%\033[0m | \033[93myellow ≥70%\033[0m | bold = max %")
    print(f"Total shown: {len(df_filtered)}\n")

    # ── Group by best timeframe + top 15 + separate heatmaps ────────────────
    best_tf = df_filtered.drop(columns=['Max_%','Strong_count','Good_count']).idxmax(axis=1)
    grouped = df_filtered.groupby(best_tf)

    print("═"*90)
    print("Grouped by strongest timeframe (showing top 15 per group)".center(90, "═"))
    print("═"*90)

    if HEATMAP_SAVE_MODE in ["all", "groups_only"]:
        for tf, group in grouped:
            if len(group) == 0:
                continue

            group_sorted = group.sort_values('Max_%', ascending=False)
            display_group = group_sorted.head(TOP_PER_GROUP)

            count_str = f"{len(group)} stocks" if len(group) <= TOP_PER_GROUP else f"{len(group)} stocks (top {TOP_PER_GROUP} shown)"
            print(f"\nBest on {tf}: {count_str}")
            print(display_group[['Max_%'] + INTERVAL_LABELS].round(1))
            print("-" * 70)

            # ── Create separate heatmap for this group (saved in day subfolder) ──
            heatmap_data = display_group.drop(columns=['Max_%', 'Strong_count', 'Good_count'])

            if not heatmap_data.empty:
                fig_height = max(5, min(18, len(heatmap_data) * 0.45))
                plt.figure(figsize=(14, fig_height))
                sns.heatmap(
                    heatmap_data,
                    annot=True,
                    fmt=".1f",
                    cmap="YlGnBu",
                    vmin=50, vmax=100,
                    linewidths=0.4,
                    annot_kws={"size": 10},
                    cbar_kws={'label': '% Up Bars', 'shrink': 0.7}
                )
                plt.title(f"{tf} – Top performers {target_date}\n(filtered ≥{MIN_THRESHOLD}%)",
                          fontsize=13, pad=12)
                plt.ylabel("Ticker")
                plt.xlabel("Timeframe")
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()

                safe_tf = tf.replace(" ", "").replace(":", "")
                heatmap_file = output_subdir / f"heatmap_{date_str}_{safe_tf}_top{len(heatmap_data)}.png"
                plt.savefig(heatmap_file, dpi=140, bbox_inches='tight')
                plt.close()
                print(f"  → Group heatmap saved: {heatmap_file}")

    print("\nGroup heatmaps completed." if HEATMAP_SAVE_MODE in ["all", "groups_only"] else "\nSkipping group heatmaps.")

    # ── Overall top heatmap (saved directly in HEATMAP_OUTPUT_DIR, not in day folder) ──
    if HEATMAP_SAVE_MODE in ["all", "overall_only"]:
        heatmap_data_all = df_filtered.drop(columns=['Max_%', 'Strong_count', 'Good_count'])

        if len(heatmap_data_all) > HEATMAP_MAX_TICKERS:
            print(f"\nToo many tickers for overall heatmap ({len(heatmap_data_all)} > {HEATMAP_MAX_TICKERS}).")
            print(f"Showing only top {HEATMAP_MAX_TICKERS} strongest performers.")
            max_vals = heatmap_data_all.max(axis=1)
            heatmap_data_all = heatmap_data_all.loc[max_vals.sort_values(ascending=False).index].iloc[:HEATMAP_MAX_TICKERS]

        if not heatmap_data_all.empty:
            fig_height = max(6, min(24, len(heatmap_data_all) * 0.38))
            plt.figure(figsize=(16, fig_height))
            sns.heatmap(
                heatmap_data_all,
                annot=True,
                fmt=".1f",
                cmap="YlGnBu",
                vmin=50, vmax=100,
                linewidths=0.3,
                annot_kws={"size": 9 if len(heatmap_data_all) > 60 else 10},
                cbar_kws={'label': '% Up Bars', 'shrink': 0.75}
            )
            plt.title(f"Overall Top Intraday % Up Bars – {target_date}\n(top {len(heatmap_data_all)} shown)",
                      fontsize=14, pad=16)
            plt.ylabel("Ticker")
            plt.xlabel("Timeframe")
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0, fontsize=9 if len(heatmap_data_all) > 60 else 10)
            plt.tight_layout()

            # ── Save overall heatmap directly in base output dir (not day subfolder) ──
            overall_file = HEATMAP_OUTPUT_DIR / f"heatmap_{date_str}_overall_top{len(heatmap_data_all)}.png"
            plt.savefig(overall_file, dpi=140, bbox_inches='tight')
            plt.close()
            print(f"\nOverall heatmap saved → {overall_file}")
        else:
            print("No data available for overall heatmap.")

    else:
        print("\nSkipping overall heatmap.")


if __name__ == "__main__":
    main()