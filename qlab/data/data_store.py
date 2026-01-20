"""
DataStore - Equities-only persistent storage for U.S. stock price data
---------------------------------------------------------------------

Purpose:
    Lightweight storage & fast retrieval for daily/weekly U.S. equities data
    (OHLCV from sources like yfinance)

Storage layout:
    ./data/
    ├── raw/        AAPL.h5, MSFT.h5, SPY.h5, ...
    └── processed/  AAPL.parquet, MSFT.parquet, SPY.parquet, ...

Features:
    - Only supports 'equities' asset class (no fx/crypto/commodities/economic)
    - HDF5 for raw archival storage (high compression)
    - Parquet for fast columnar queries
    - Polars in-memory cache for repeated same-ticker access
    - Normalizes yfinance multi-index → flat columns
    - Enforces UTC datetime index
    - Query fallback: cache → parquet → hdf5

Typical usage:
    store = DataStore()
    store.write('AAPL', df)                     # asset_class is assumed 'equities'
    recent = store.query('AAPL', '2025-01-01', '2026-01-19')
"""
import pandas as pd
import polars as pl
import os
from datetime import datetime
import pyarrow.parquet as pq
import pyarrow as pa

class DataStore:
    def __init__(self, storage_path="./data"):
        self.storage_path = storage_path
        self.asset_class = "equities"  # fixed

        self.raw_path      = os.path.join(storage_path, "raw",      "equities")
        self.processed_path = os.path.join(storage_path, "processed", "equities")
        os.makedirs(self.raw_path,      exist_ok=True)
        os.makedirs(self.processed_path, exist_ok=True)

        self.cache = {}  # {ticker: pl.DataFrame}

    def _normalize_columns(self, df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        return df

    def write(self, ticker, df):
        """Write OHLCV DataFrame for one equity ticker"""
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        df = self._normalize_columns(df)

        # HDF5 (raw)
        hdf_path = os.path.join(self.raw_path, f"{ticker}.h5")
        df.to_hdf(hdf_path, key=ticker, mode='w', complevel=9, complib='blosc')

        # Parquet (processed)
        parquet_path = os.path.join(self.processed_path, f"{ticker}.parquet")
        table = pa.Table.from_pandas(df)
        pq.write_table(table, parquet_path, compression='snappy')

        # Polars cache
        df_pl = pl.from_pandas(df.reset_index())
        if 'index' in df_pl.columns:
            df_pl = df_pl.rename({'index': 'Date'})
        df_pl = df_pl.with_columns(pl.col("Date").cast(pl.Datetime("us", "UTC")))
        self.cache[ticker] = df_pl

    def query(self, ticker, start, end):
        """Query date range for one equity ticker"""
        start_dt = pd.to_datetime(start).tz_localize('UTC')
        end_dt   = pd.to_datetime(end).tz_localize('UTC')

        # 1. Cache hit
        if ticker in self.cache:
            df_pl = self.cache[ticker].filter(
                (pl.col("Date") >= start_dt) & (pl.col("Date") <= end_dt)
            )
            if not df_pl.is_empty():
                return df_pl.to_pandas().set_index("Date")

        # 2. Parquet
        parquet_path = os.path.join(self.processed_path, f"{ticker}.parquet")
        if os.path.exists(parquet_path):
            df_pl = pl.read_parquet(parquet_path).filter(
                (pl.col("Date") >= start_dt) & (pl.col("Date") <= end_dt)
            )
            if not df_pl.is_empty():
                self.cache[ticker] = df_pl  # cache full dataset
                return df_pl.to_pandas().set_index("Date")

        # 3. HDF5 fallback
        hdf_path = os.path.join(self.raw_path, f"{ticker}.h5")
        if os.path.exists(hdf_path):
            df = pd.read_hdf(hdf_path, key=ticker)
            df = self._normalize_columns(df)
            df = df.loc[start_dt:end_dt]
            if not df.empty:
                self.write(ticker, df)  # refresh parquet & cache
                return df

        raise ValueError(f"No data found for {ticker}")