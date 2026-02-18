# src/qlab/data/fetch.py
"""
Unified Yahoo Finance fetcher – prices + fundamentals (balance sheet & income statement).

Production-oriented features:
- Incremental updates with deduplication and conflict resolution
- Bulk multi-ticker downloads for speed
- Zstandard-compressed Parquet storage
- Separate folders: prices/{interval} and fundamentals/{kind}
- Configurable skip logic based on file age
- Polite rate limiting + randomized delays
- Basic data quality logging (gaps, negatives, short history)

Goal: Fast, reliable ingestion layer for alpha research & backtesting pipelines.
 
Usage examples:
    fetcher = DataFetcher("data/yfinance")
    fetcher.get("SPY", interval="1d")                           # prices only
    fetcher.get("AAPL", fetch_fundamentals=True)                # prices + fundamentals
    fetcher.fetch_batch(1, 30, fetch_fundamentals=True)         # batch mode
"""

from __future__ import annotations

import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path
from ratelimit import limits, sleep_and_retry
from tqdm.auto import tqdm
import time
import random
import warnings
from typing import Dict

warnings.filterwarnings("ignore", category=FutureWarning)  # suppress yfinance noise

class FetchConfig:
    """Central configuration – easy to tune or override."""
    SAVE_ROOT: str = "market_data/yfinance"
    COMPRESSION: str = "zstd"
    SKIP_PRICES_DAYS: int = 3
    SKIP_FUNDAMENTALS_DAYS: int = 45
    BATCH_SUB_SIZE: int = 80
    POLIT_DELAY_BASE: float = 1.8
    POLIT_DELAY_JITTER: float = 2.2


class YahooFinanceSource:
    """Low-level Yahoo Finance interface with rate limiting."""

    INTRADAY_INTERVALS = {"1m", "2m", "3m", "5m", "15m", "30m", "60m", "90m", "1h"}

    @sleep_and_retry
    @limits(calls=2, period=1)
    def get_prices(
            self,
            tickers: Union[str, list[str]],
            start: Optional[str] = None,
            period: Optional[str] = None,
            interval: str = "1d",
            fallback_single: bool = True,          # new: try singles if bulk fails
            min_rows_expected: int = 10,           # new: basic quality gate
        ) -> Optional[pd.DataFrame]:
            """
            Fetch OHLCV (and Adj Close if available) for one or multiple tickers.
            
            Returns:
                MultiIndex DataFrame with group_by='ticker' (ticker as level 0)
                or None if failed / empty.
            """
            if isinstance(tickers, str):
                tickers = [tickers.strip().upper()]
            else:
                tickers = [t.strip().upper() for t in tickers if t.strip()]

            if not tickers:
                return None

            tickers_str = " ".join(tickers)

            kwargs = {
                "tickers": tickers_str,
                "interval": interval,
                "progress": False,
                "auto_adjust": False,           # keep raw + Adj Close separate
                "threads": True if len(tickers) > 1 else False,
                "timeout": 25,
                "group_by": "ticker",           # ← KEY: ticker as top level
                # "multi_level_index": True,    # usually default now; can toggle if needed
            }

            if period:
                kwargs["period"] = period
            elif start:
                kwargs["start"] = start
            else:
                kwargs["period"] = "max"

            try:
                data = yf.download(**kwargs)

                if data is None or data.empty:
                    raise ValueError("Empty DataFrame returned")

                # ── Quality check ───────────────────────────────────────
                if len(data) < min_rows_expected:
                    print(f"Warning: fetched data too short ({len(data)} rows) for {tickers_str[:60]}...")
                    return None

                # Ensure expected columns exist (at least for first ticker)
                first_ticker = tickers[0]
                if first_ticker not in data.columns.levels[0]:
                    print(f"Warning: ticker {first_ticker} not found in level 0 (group_by='ticker')")
                    # Fallback: try level 1 (older yfinance behavior)
                    if first_ticker in data.columns.levels[1]:
                        data = data.xs(first_ticker, level=1, axis=1)  # rare now

                return data

            except Exception as e:
                print(f"Bulk price fetch failed ({tickers_str[:60]}...): {e}")

                if not fallback_single or len(tickers) <= 1:
                    return None

                # ── Fallback: one-by-one (more reliable under throttling) ──
                print(f"  → Falling back to single-ticker downloads ({len(tickers)} tickers)")
                results = {}

                for ticker in tickers:
                    try:
                        single = yf.download(
                            ticker,
                            period=period,
                            start=start,
                            interval=interval,
                            progress=False,
                            auto_adjust=False,
                            timeout=15,
                        )
                        if not single.empty and len(single) >= min_rows_expected:
                            results[ticker] = single
                        time.sleep(0.7 + random.uniform(0, 1.2))  # polite jitter
                    except Exception as ex:
                        print(f"    Single fetch failed {ticker}: {ex}")

                if not results:
                    return None

                # Combine into multi-ticker format (mimic group_by='ticker')
                combined = pd.concat(
                    {ticker: df for ticker, df in results.items()},
                    axis=1
                ).swaplevel(axis=1).sort_index(axis=1)  # ticker first

                return combined if not combined.empty else None

    def get_fundamentals(self, ticker: str) -> Dict[str, pd.DataFrame]:
        try:
            t = yf.Ticker(ticker)
            return {
                "balance_sheet": t.balance_sheet,
                "quarterly_balance_sheet": t.quarterly_balance_sheet,
                "income_statement": t.financials,
                "quarterly_income_statement": t.quarterly_financials,
            }
        except Exception as e:
            print(f"Fundamentals fetch failed for {ticker}: {e}")
            return {}


class DataFetcher:
    def __init__(self, save_dir: str = FetchConfig.SAVE_ROOT):
        self.source = YahooFinanceSource()
        self.cfg = FetchConfig()
        self.save_root = Path(save_dir).resolve()
        self.save_root.mkdir(parents=True, exist_ok=True)
        self.last_fetch_week = None

    def _get_current_week(self) -> tuple[int, int]:
        return datetime.now().isocalendar()[:2]

    def _should_fetch_this_week(self) -> bool:
        current = self._get_current_week()
        if self.last_fetch_week != current:
            self.last_fetch_week = current
            return True
        return False

    # ── File paths ───────────────────────────────────────────────────────────────

    def _get_price_path(self, ticker: str, interval: str) -> Path:
        folder = self.save_root / "prices" / interval.lower().replace(" ", "")
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{ticker.upper()}.parquet"

    def _get_fundamentals_path(self, ticker: str, kind: str) -> Path:
        folder = self.save_root / "fundamentals" / kind
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{ticker.upper()}.parquet"

    # ── Core single-ticker fetch ───────────────────────────────────────────────

    def get(
        self,
        ticker: str,
        interval: str = "1d",
        start: str | None = None,
        period: str | None = None,
        save: bool = True,
        force_full: bool = False,
        skip_existing_days: int | None = None,
        fetch_fundamentals: bool = False,
    ) -> pd.DataFrame | None:
        """Fetch price history (incremental by default) + optionally fundamentals."""
        ticker = ticker.strip().upper()
        path = self._get_price_path(ticker, interval)
        skip_days = skip_existing_days if skip_existing_days is not None else self.cfg.SKIP_PRICES_DAYS

        # Quick skip if recent
        if path.exists() and not force_full:
            age_days = (time.time() - path.stat().st_mtime) / 86400
            if age_days < skip_days:
                try:
                    df = pd.read_parquet(path)
                    if fetch_fundamentals:
                        self.get_fundamentals(ticker, save=save, force=force_full)
                    return df
                except Exception:
                    pass

        # Smart period defaults for intraday
        if interval in self.source.INTRADAY_INTERVALS and not period and not start:
            period = "7d" if interval == "1m" else "60d"

        old_df = pd.DataFrame()
        if path.exists() and not force_full:
            try:
                old_df = pd.read_parquet(path)
            except Exception as e:
                print(f"Read error {path}: {e}")

        last_date = old_df.index.max() if not old_df.empty else None

        if last_date and not self._should_fetch_this_week() and not force_full:
            if (datetime.now() - last_date).days < 5:
                if fetch_fundamentals:
                    self.get_fundamentals(ticker, save=save)
                return old_df

        new_df = self.source.get_prices(ticker, start=start, period=period, interval=interval)

        if new_df is None or new_df.empty:
            if fetch_fundamentals:
                self.get_fundamentals(ticker, save=save)
            return old_df if not old_df.empty else None

        # Merge + clean
        if not old_df.empty:
            combined = pd.concat([old_df, new_df]).sort_index()
            before = len(combined)
            combined = combined[~combined.index.duplicated(keep="last")]
            if combined.index.duplicated().any():
                combined = combined.loc[combined.groupby(combined.index)["Volume"].idxmax()]
            if len(combined) < before:
                print(f"{ticker} prices: cleaned {before - len(combined)} rows")
        else:
            combined = new_df

        if save:
            combined.to_parquet(path, compression=self.cfg.COMPRESSION)
            print(f"Saved/updated prices: {path} ({len(combined)} rows)")

        if fetch_fundamentals:
            self.get_fundamentals(ticker, save=save, force=force_full)

        return combined

    def get_fundamentals(
        self,
        ticker: str,
        save: bool = True,
        force: bool = False,
        skip_existing_days: int | None = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch and save balance sheet + income statement (annual & quarterly)."""
        ticker = ticker.strip().upper()
        results = {}
        skip_days = skip_existing_days if skip_existing_days is not None else self.cfg.SKIP_FUNDAMENTALS_DAYS

        for kind in [
            "balance_sheet",
            "quarterly_balance_sheet",
            "income_statement",
            "quarterly_income_statement",
        ]:
            path = self._get_fundamentals_path(ticker, kind)

            if path.exists() and not force:
                age_days = (time.time() - path.stat().st_mtime) / 86400
                if age_days < skip_days:
                    try:
                        df = pd.read_parquet(path)
                        results[kind] = df
                        continue
                    except Exception:
                        pass

            data = self.source.get_fundamentals(ticker)
            if kind not in data or data[kind].empty:
                continue

            df = data[kind].T  # wide format: periods as rows, items as columns
            df.index.name = "date" if "quarterly" not in kind else "quarter_end"

            # Try to ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except:
                    pass

            df = df.sort_index(ascending=False)  # most recent first

            # Basic quality flags
            if (df < 0).any().any():
                print(f"Warning: {ticker} {kind} contains negative values")

            results[kind] = df

            if save:
                df.to_parquet(path, compression=self.cfg.COMPRESSION)
                print(f"Saved {kind}: {path} ({len(df)} periods × {len(df.columns)} items)")

        return results

    def fetch_batch(
        self,
        start_batch: int,
        end_batch: int,
        interval: str = "1d",
        batch_dir: str = "market_data/yfinance/batches",
        fetch_fundamentals: bool = False,
        skip_existing_days_prices: int | None = None,
        skip_existing_days_fund: int | None = None,
    ):
        """Batch process many tickers – prices + optionally fundamentals."""
        batch_dir = Path(batch_dir)
        skip_p = skip_existing_days_prices or self.cfg.SKIP_PRICES_DAYS
        skip_f = skip_existing_days_fund or self.cfg.SKIP_FUNDAMENTALS_DAYS

        total_prices = total_fund = total_skipped = total_failed = 0

        for batch_num in range(start_batch, end_batch + 1):
            batch_file = batch_dir / f"batch_{batch_num:03d}.txt"
            if not batch_file.exists():
                print(f"Batch file missing: {batch_file}")
                continue

            print(f"\n{'═'*60} Batch {batch_num:03d} {'═'*60}")

            with open(batch_file) as f:
                tickers = [line.strip().upper() for line in f if line.strip()]

            to_fetch_prices = []
            to_fetch_fund = []

            for t in tickers:
                p_price = self._get_price_path(t, interval)
                need_price = True
                if p_price.exists() and not skip_p == 0:
                    age = (time.time() - p_price.stat().st_mtime) / 86400
                    if age < skip_p:
                        need_price = False

                if need_price:
                    to_fetch_prices.append(t)
                else:
                    total_skipped += 1

                if fetch_fundamentals:
                    p_fund = self._get_fundamentals_path(t, "balance_sheet")  # proxy
                    need_fund = True
                    if p_fund.exists() and not skip_f == 0:
                        age = (time.time() - p_fund.stat().st_mtime) / 86400
                        if age < skip_f:
                            need_fund = False
                    if need_fund:
                        to_fetch_fund.append(t)

            if not to_fetch_prices and (not fetch_fundamentals or not to_fetch_fund):
                print("All files recent → skipping batch")
                continue

            print(f"→ Prices to fetch: {len(to_fetch_prices)} | Fundamentals: {len(to_fetch_fund)}")

            # ── Price batch fetch ───────────────────────────────────────
            success_p = failed_p = 0
            for i in tqdm(range(0, len(to_fetch_prices), self.cfg.BATCH_SUB_SIZE), desc="Prices"):
                sub = to_fetch_prices[i : i + self.cfg.BATCH_SUB_SIZE]
                try:
                    data = self.source.get_prices(sub, interval=interval)
                    if data is None or data.empty:
                        failed_p += len(sub)
                        continue

                    for ticker in sub:
                        if ticker not in data.columns.levels[0]:
                            failed_p += 1
                            continue
                        df = data[ticker].dropna(how="all")
                        if df.empty:
                            failed_p += 1
                            continue
                        path = self._get_price_path(ticker, interval)
                        self._save_incremental_price(path, df)
                        success_p += 1

                except Exception as e:
                    print(f"Price sub-batch failed ({len(sub)} tickers): {e}")
                    failed_p += len(sub)

                time.sleep(self.cfg.POLIT_DELAY_BASE + random.uniform(0, self.cfg.POLIT_DELAY_JITTER))

            total_prices += success_p
            total_failed += failed_p

            

        # ── Fundamentals (one-by-one – Yahoo doesn't bulk fundamentals) ──
        if fetch_fundamentals and to_fetch_fund:
            print(f"\nStarting fundamentals fetch for {len(to_fetch_fund)} tickers...")
            print("This may take several minutes – Yahoo rate-limits single-ticker calls.")
            
            success_f = failed_f = 0
            for ticker in tqdm(to_fetch_fund, desc="Fundamentals"):
                try:
                    results = self.get_fundamentals(ticker, save=True, force=False)
                    if results:  # if any data was returned/saved
                        success_f += 1
                        print(f"  Saved fundamentals for {ticker}")
                    else:
                        print(f"  No fundamentals data for {ticker}")
                except Exception as e:
                    print(f"  Fundamentals failed for {ticker}: {e}")
                    failed_f += 1
                time.sleep(1.2 + random.uniform(0, 1.8))

            total_fund += success_f
            total_failed += failed_f

            print(f"Batch {batch_num:03d} summary:")
            print(f"  Prices: {success_p} saved | {failed_p} failed | {total_skipped} skipped")
            print(f"  Fundamentals: {success_f} saved | {failed_f} failed")

        print(f"\nOverall: Prices {total_prices} | Fundamentals {total_fund} | Skipped {total_skipped} | Failed {total_failed}")

    def _save_incremental_price(self, path: Path, new_df: pd.DataFrame):
        """Shared incremental save logic for prices."""
        if path.exists():
            try:
                old_df = pd.read_parquet(path)
                combined = pd.concat([old_df, new_df]).sort_index()
                before = len(combined)
                combined = combined[~combined.index.duplicated(keep="last")]
                if combined.index.duplicated().any():
                    combined = combined.loc[combined.groupby(combined.index)["Volume"].idxmax()]
                if len(combined) < before:
                    print(f"  Cleaned {before - len(combined)} rows in {path.name}")
            except Exception:
                combined = new_df
        else:
            combined = new_df

        combined.to_parquet(path, compression=self.cfg.COMPRESSION)


# ── CLI for batch runs ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Yahoo Finance batch fetcher – prices + fundamentals",
        epilog="Examples:\n"
               "  python fetch.py 1                     # batch 1, prices only\n"
               "  python fetch.py 1 --fundamentals      # batch 1, prices + fundamentals\n"
               "  python fetch.py 1 5 --fundamentals    # batches 1–5 with fundamentals\n"
               "  python fetch.py 1 --fundamentals --force  # force full re-fetch"
    )

    parser.add_argument("start", type=int, nargs="?", default=1, help="Start batch")
    parser.add_argument("end",   type=int, nargs="?", default=None, help="End batch (default = start)")
    parser.add_argument("--interval", default="1d", help="Data interval")
    parser.add_argument("--fundamentals", action="store_true", help="Fetch balance & income statements")
    parser.add_argument("--force", action="store_true", help="Force full re-fetch (ignore skip logic)")
    parser.add_argument("--skip-prices", type=int, default=None, help="Override price skip days")
    parser.add_argument("--skip-fund",   type=int, default=None, help="Override fundamentals skip days")

    args = parser.parse_args()

    if args.end is None:
        args.end = args.start

    # Determine effective skip values
    skip_prices = 0 if args.force else (args.skip_prices or FetchConfig.SKIP_PRICES_DAYS)
    skip_fund   = 0 if args.force else (args.skip_fund   or FetchConfig.SKIP_FUNDAMENTALS_DAYS)

    print(f"Fetching batches {args.start}–{args.end}")
    print(f"  Interval:     {args.interval}")
    print(f"  Fundamentals: {'yes' if args.fundamentals else 'no'}")
    print(f"  Skip prices:  {skip_prices} days")
    print(f"  Skip fund:    {skip_fund} days")
    if args.force:
        print("  → FORCE mode enabled")

    fetcher = DataFetcher()
    fetcher.fetch_batch(
        start_batch=args.start,
        end_batch=args.end,
        interval=args.interval,
        fetch_fundamentals=args.fundamentals,
        skip_existing_days_prices=skip_prices,
        skip_existing_days_fund=skip_fund,
    )