"""
Data Quality Checker for Stock OHLCV Data
-----------------------------------------

Purpose:
    Analyzes and fixes quality issues in fetched stock price data from sources like yfinance.
    Checks completeness, accuracy, timeliness, and outliers.

Key Features:
    - Completeness: Detects missing trading days, NaN streaks, and missing values.
    - Accuracy: Validates non-positive prices/volume and suspicious split adjustments.
    - Timeliness: Checks if data is up-to-date (within 3 days of latest trading day).
    - Outliers: Identifies extreme daily returns (>6 std devs).
    - Fixing: Deduplicates, forward-fills limited NaNs, drops all-NaN rows.
    - Uses NYSE calendar for expected trading days.

Dependencies:
    - pandas, numpy
    - pandas_market_calendars
    - DataFetcher (for data retrieval)

Usage:
    checker = QualityChecker()
    report = checker.check('AAPL', start_date='2025-01-01')
    fixed_df = checker.fix('AAPL', start_date='2025-01-01', forward_fill=True)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pandas_market_calendars import get_calendar
from data_fetcher import DataFetcher

class QualityChecker:
    def __init__(self, fetcher=None):
        self.fetcher = fetcher or DataFetcher()
        self.nyse = get_calendar('NYSE')

    def check(self, ticker, start_date=None):
        """
        Check data quality for a ticker.
        Optional start_date limits the analysis period.
        """
        try:
            df = self.fetcher.get(
                ticker,
                start=start_date or "1900-01-01",  # very early fallback
                use_earliest_if_unavailable=True
            )
        except Exception as e:
            return {"error": str(e), "status": "fetch_failed"}

        if df is None or df.empty:
            return {"error": "No data returned", "status": "empty"}

        # Ensure UTC datetime index
        if df.index.tz is None:
            df.index = pd.to_datetime(df.index).tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')

        report = {
            "ticker": ticker,
            "status": "ok",
            "row_count": len(df),
            "date_range": {
                "first": str(df.index.min().date()),
                "last": str(df.index.max().date())
            },
            "completeness": {},
            "accuracy": {},
            "timeliness": {},
            "outliers": [],
            "warnings": []
        }

        # ── Completeness ────────────────────────────────────────────────────────
        start = df.index.min()
        end = df.index.max()
        expected = self.nyse.valid_days(start_date=start.date(), end_date=end.date())
        present_dates = df.index.normalize()  # remove time part if any
        missing = expected[~expected.isin(present_dates)]
        
        report["completeness"] = {
            "missing_trading_days": len(missing),
            "missing_days_list": [str(d.date()) for d in missing[-10:]],  # last 10
            "missing_values": df.isna().sum().to_dict(),
            "longest_nan_streak": self._longest_nan_streak(df['Close']) if 'Close' in df else 0
        }

        # ── Accuracy ────────────────────────────────────────────────────────────
        price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        vol_cols = ['Volume']

        invalid_prices = {}
        for col in price_cols:
            if col in df:
                invalid = df[col] <= 0
                if invalid.any():
                    invalid_prices[col] = invalid.sum()

        invalid_volume = (df['Volume'] <= 0).sum() if 'Volume' in df else 0

        report["accuracy"] = {
            "invalid_prices": invalid_prices,
            "invalid_volume": invalid_volume,
            "suspicious_splits": self._check_split_factor_sanity(df)
        }

        # ── Timeliness ──────────────────────────────────────────────────────────
        today = pd.Timestamp.now(tz='UTC').normalize()
        last_expected = self.nyse.valid_days(
            start_date=(today - pd.Timedelta(days=60)).date(),
            end_date=today.date()
        )[-1] if not self.nyse.valid_days(
            start_date=today.date(), end_date=today.date()
        ).empty else today - pd.Timedelta(days=1)

        latest = df.index.max().normalize()
        days_behind = (last_expected - latest).days if latest < last_expected else 0

        report["timeliness"] = {
            "latest_date": str(latest.date()),
            "days_behind_current": days_behind,
            "is_up_to_date": days_behind <= 3
        }

        # ── Outliers ────────────────────────────────────────────────────────────
        if 'Close' in df and len(df) >= 30:
            ret = df['Close'].pct_change().dropna()
            if len(ret) >= 20:
                rolling_std = ret.rolling(60, min_periods=20).std()
                zscore = (ret - ret.rolling(60, min_periods=20).mean()) / rolling_std
                extreme = ret[np.abs(zscore) > 6].index  # stricter threshold
                report["outliers"] = [str(d.date()) for d in extreme[-8:]]

        return report

    def fix(self, ticker, start_date=None, forward_fill=True, adjust_splits=True):
        df = self.fetcher.get(
            ticker,
            start=start_date or "1900-01-01",
            use_earliest_if_unavailable=True
        )
        if df is None or df.empty:
            raise ValueError(f"No data to fix for {ticker}")

        df = df.copy()

        # Timezone
        if df.index.tz is None:
            df.index = pd.to_datetime(df.index).tz_localize('UTC')

        # Deduplicate + keep last (safer than first)
        df = df.loc[~df.index.duplicated(keep='last')]

        # Warning: do NOT blindly adjust OHLC with Adj Close / Close ratio
        # yfinance already incorporates splits & dividends in Adj Close
        # If you really need split-adjusted non-adj prices → use raw data + split events

        if forward_fill:
            df = df.ffill(limit=5)  # limit to avoid filling years of missing data

        df = df.dropna(how='all')

        return df

    # Helpers ────────────────────────────────────────────────────────────────────
    @staticmethod
    def _longest_nan_streak(series):
        """Max consecutive NaNs in a series"""
        if series.isna().all():
            return len(series)
        return series.isna().astype(int).groupby(series.notna().cumsum()).sum().max()

    @staticmethod
    def _check_split_factor_sanity(df):
        """Quick sanity check on implied split factors"""
        if 'Adj Close' not in df or 'Close' not in df:
            return {}
        factor = df['Adj Close'] / df['Close']
        suspicious = factor[factor < 0.1]  # sudden 10x+ adjustment
        if not suspicious.empty:
            return {"large_adjustments": len(suspicious), "dates": suspicious.index.strftime('%Y-%m-%d').tolist()[:5]}
        return {}