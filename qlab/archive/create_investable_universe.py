# qlab/universe/create_investable_universe.py
"""
Point-in-time investable universe construction.

Goals:
- Apply realistic liquidity & quality filters month-by-month (PIT)
- Avoid look-ahead bias → only use information available at each rebalance date
- Output long-format Parquet: one row per (date, ticker)
- Configurable thresholds + detailed logging for transparency

Typical usage:
    python create_investable_universe.py --start 20100101 --end 20251231
    python create_investable_universe.py --year 2024 --force
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse
import logging
from typing import List, Optional

# ── Configuration ────────────────────────────────────────────────────────────────

class UniverseConfig:
    """Central tuning parameters — mirror style from fetch.py"""

    DATA_ROOT          = Path("market_data/yfinance")
    PRICES_SUBDIR      = "prices/1d"
    UNIVERSE_SUBDIR    = "universes"
    OUTPUT_FORMAT      = "parquet"          # or "csv" if you prefer

    # Core liquidity / quality filters (very common institutional defaults)
    MIN_ADV_USD        = 1_000_000         # $1M average daily dollar volume
    MIN_PRICE_USD      = 5.0               # avoid pennies
    MIN_TRAILING_MONTHS = 12               # at least 1 year of history
    MAX_DAYS_SINCE_IPO = None              # None = no restriction; set e.g. 365*2

    # Exclusions
    EXCLUDE_ETF        = True
    EXCLUDE_ADR        = True              # rough heuristic
    EXCLUDE_FINANCIALS = False             # often excluded in some strategies

    # Rebalance frequency
    REBALANCE_FREQ     = "ME"              # month-end; can be "QE", "YE", "W-FRI", etc.

    # Logging
    LOG_LEVEL          = logging.INFO


# ── Helpers ──────────────────────────────────────────────────────────────────────

def setup_logging():
    logging.basicConfig(
        level=UniverseConfig.LOG_LEVEL,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def get_all_price_files() -> List[Path]:
    price_dir = UniverseConfig.DATA_ROOT / UniverseConfig.PRICES_SUBDIR
    return sorted(price_dir.glob("*.parquet"))


def ticker_from_path(p: Path) -> str:
    return p.stem.upper()


def load_prices_for_ticker(path: Path) -> pd.DataFrame:
    """Load single ticker parquet → keep essential columns"""
    try:
        df = pd.read_parquet(path, columns=["Open", "High", "Low", "Close", "Volume", "Adj Close"])
        df = df[["Adj Close", "Close", "Volume"]].dropna(how="all")  # keep most important
        df["Adj Close"] = df["Adj Close"].ffill()                   # simple forward-fill
        return df
    except Exception as e:
        logging.warning(f"Failed to read {path.name}: {e}")
        return pd.DataFrame()


def compute_monthly_metrics_and_quality(df: pd.DataFrame, as_of_date: pd.Timestamp) -> dict:
    """
    Returns {
        'valid': bool,
        'adv_usd': float,
        'price': float,
        'months_history': float,
        'n_days': int,
        'quality_pass': bool,
        'quality_issues': list[str]   # e.g. ['negative_price', 'extreme_jump']
    }
    """

    past = df.loc[:as_of_date].copy()
    if len(past) < 20:
        return {"valid": False, "quality_pass": False, "quality_issues": ["too_short_history"]}

    recent = past.tail(42)  # ~2 months
    if len(recent) < 21:
        return {"valid": False, "quality_pass": False, "quality_issues": ["insufficient_recent_data"]}

    # ── Existing liquidity calcs ───────────────────────────────────────
    adv_shares = recent["Volume"].mean()
    adv_usd    = (recent["Close"] * recent["Volume"]).mean()
    price      = past["Adj Close"].iloc[-1]
    months_history = (as_of_date - past.index.min()).days / 30.437

    issues = []

    # ── Quality checks ─────────────────────────────────────────────────
    # 1. Negative / zero price
    if price <= 0:
        issues.append("non_positive_price")

    # 2. Negative volume ever
    if (recent["Volume"] < 0).any():
        issues.append("negative_volume")

    # 3. Zero volume on days with price change
    changed = recent["Adj Close"].diff().abs() > 0
    zero_vol_on_change = (recent["Volume"] == 0) & changed
    if zero_vol_on_change.any():
        issues.append("zero_volume_on_price_change")

    # 4. Extreme single-day return
    daily_ret = recent["Adj Close"].pct_change()
    if (daily_ret.abs() > 1.0).any():  # >100% move
        issues.append("extreme_price_jump")

    # 5. Long stale period (unchanged adj close)
    unchanged_streak = (recent["Adj Close"].diff() == 0).astype(int).groupby((recent["Adj Close"].diff() != 0).cumsum()).cumsum()
    if unchanged_streak.max() > 15:
        issues.append("long_stale_price")

    # 6. Too few actual trading days recently
    active_days = (recent["Volume"] > 0).sum()
    if active_days < 15:
        issues.append("insufficient_active_days")

    quality_pass = len(issues) == 0

    return {
        "valid": True,                     # keep old logic
        "adv_usd": adv_usd,
        "price": price,
        "months_history": months_history,
        "n_days": len(past),
        "quality_pass": quality_pass,
        "quality_issues": issues
    }

def is_likely_etf(ticker: str) -> bool:
    """Very rough heuristic — improve later with external list if needed"""
    etf_indicators = [" ETF", "ETN", " ETP", " TRUST", "^", "=", "-"]
    return any(ind in ticker for ind in etf_indicators) or len(ticker) > 5 and ticker[-1].isdigit()


def is_likely_adr(ticker: str) -> bool:
    """Rough — many ADRs end in letters but not perfect"""
    return len(ticker) > 4 and ticker[-1].isalpha() and ticker[-2].isalpha()


def filter_ticker(ticker: str, metrics: dict, cfg: UniverseConfig) -> bool:
    if not metrics["valid"]:
        return False
    
    if not metrics.get("quality_pass", False):
        # Optional: log why rejected
        # logger.debug(f"{ticker} failed quality: {metrics['quality_issues']}")
        return False

    if cfg.MIN_PRICE_USD is not None and metrics["price"] < cfg.MIN_PRICE_USD:
        return False

    if cfg.MIN_ADV_USD is not None and metrics["adv_usd"] < cfg.MIN_ADV_USD:
        return False

    if metrics["months_history"] < cfg.MIN_TRAILING_MONTHS:
        return False

    if cfg.EXCLUDE_ETF and is_likely_etf(ticker):
        return False

    if cfg.EXCLUDE_ADR and is_likely_adr(ticker):
        return False

    # Add more filters later (sector, market cap decile, etc.)
    return True


# ── Main logic ───────────────────────────────────────────────────────────────────

def build_universe(start_date: str, end_date: str, force: bool = False):
    logger = setup_logging()
    cfg = UniverseConfig()

    out_dir = cfg.DATA_ROOT / cfg.UNIVERSE_SUBDIR
    out_dir.mkdir(exist_ok=True, parents=True)

    price_files = get_all_price_files()
    logger.info(f"Found {len(price_files)} price files")

    # Generate rebalance dates
    dates = pd.date_range(start_date, end_date, freq=cfg.REBALANCE_FREQ, inclusive="both")
    logger.info(f"Generating universe for {len(dates)} rebalance dates")

    all_rows = []

    for as_of_date in dates:
        logger.info(f"Processing {as_of_date.date()} ...")

        valid_tickers = []
        total_raw = len(price_files)

        for path in price_files:
            ticker = ticker_from_path(path)

            df = load_prices_for_ticker(path)
            if df.empty:
                continue

            metrics = compute_monthly_metrics_and_quality(df, as_of_date)

            if filter_ticker(ticker, metrics, cfg):
                valid_tickers.append({
                    "date": as_of_date,
                    "ticker": ticker,
                    "adv_usd": round(metrics["adv_usd"], 0),
                    "price_close": round(metrics["price"], 2),
                    "months_history": round(metrics["months_history"], 1),
                    "n_days": metrics["n_days"],
                })

        if valid_tickers:
            month_df = pd.DataFrame(valid_tickers)
            count = len(month_df)
            logger.info(f"  → {count:4d} tickers survive filters ({count/total_raw:.1%} of raw)")
            all_rows.append(month_df)

    if not all_rows:
        logger.warning("No valid universes generated")
        return

    final = pd.concat(all_rows, ignore_index=True)
    final["date"] = pd.to_datetime(final["date"]).dt.strftime("%Y-%m-%d")  # clean string

    # Sort & save
    final = final.sort_values(["date", "ticker"])
    out_path = out_dir / f"investable_universe_pit_{start_date[:4]}_{end_date[:4]}.parquet"
    final.to_parquet(out_path, compression="zstd", index=False)

    logger.info(f"Saved: {out_path}")
    logger.info(f"Total rows: {len(final):,}")
    logger.info(f"Unique tickers ever: {final['ticker'].nunique():,}")
    logger.info(f"Median tickers per month: {final.groupby('date').size().median():.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build point-in-time investable universe")
    parser.add_argument("--start", default="20100101", help="Start date YYYYMMDD")
    parser.add_argument("--end",   default="20251231", help="End date YYYYMMDD")
    parser.add_argument("--year",  type=int, help="Shortcut: process single year")
    parser.add_argument("--force", action="store_true", help="Overwrite if exists")

    args = parser.parse_args()

    if args.year:
        args.start = f"{args.year}0101"
        args.end   = f"{args.year}1231"

    build_universe(args.start, args.end, force=args.force)