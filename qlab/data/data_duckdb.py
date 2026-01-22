# qlab_duckdb.py
# ──────────────────────────────────────────────────────────────────────────────
# DuckDB setup for QLab: Register ALL Parquet files as virtual views
# Works with your existing folder structure: data/yfinance/<interval>/*.parquet
# ──────────────────────────────────────────────────────────────────────────────

import duckdb
from pathlib import Path
import os

# ── Configuration ───────────────────────────────────────────────────────────────
BASE_DATA_DIR = Path("data/yfinance")           # Your root folder
INTERVALS = [
    '1m', '2m', '3m', '5m', '7m', '15m', '30m',
    '60m', '120m', '240m',                      # intraday
    '1d', '1wk', '1mo'                          # daily+ (add more if you have them)
]

# Where to save the DuckDB database file (optional - for persistence)
DB_PATH = Path("qlab/data/qlab.duckdb")              # Will be created if it doesn't exist

# ── Helper: Clean symbol name from filename ────────────────────────────────────
def clean_symbol(filename: str) -> str:
    """Turn 'AAPL.parquet' → 'AAPL' """
    return Path(filename).stem.upper()          # removes .parquet and makes uppercase

# ── Main DuckDB connection & setup ─────────────────────────────────────────────
def setup_qlab_duckdb(db_path: Path = DB_PATH, read_only=False):
    """
    Connect to DuckDB and register ALL your Parquet files as views.
    - read_only=True: faster startup if you won't write data
    """
    # Connect (create file if it doesn't exist)
    con = duckdb.connect(str(db_path), read_only=read_only)
    
    print(f"Connected to DuckDB: {db_path.resolve()}")
    
    # Optional: Increase threads for faster queries on large data
    con.execute("SET threads TO 8;")  # Adjust to your CPU cores (e.g., 12, 16)
    
    # Register views for EVERY interval
    for interval in INTERVALS:
        parquet_folder = BASE_DATA_DIR / interval
        if not parquet_folder.exists() or not any(parquet_folder.glob("*.parquet")):
            print(f"Skipping {interval} — no Parquet files found")
            continue
        
        view_name = f"bars_{interval.replace('m', 'min').replace('d', 'day').replace('wk', 'week').replace('mo', 'month')}"
        
        # Use filename=true to get the source file name as a column
        query = f"""
        CREATE OR REPLACE VIEW {view_name} AS
        SELECT
            '{interval}'                                      AS interval,
            regexp_replace(filename, '.parquet$', '')         AS symbol,   -- or use clean_symbol(filename)
            "Datetime"                                            AS ts,       -- your datetime index
            "Open"::DOUBLE                                    AS open,
            High::DOUBLE                                      AS high,
            Low::DOUBLE                                       AS low,
            "Close"::DOUBLE                                   AS close,
            Volume::BIGINT                                    AS volume,
            filename                                          AS source_file
        FROM read_parquet(
            '{parquet_folder}/*.parquet',
            filename = true
        )
        """
        
        try:
            con.execute(query)
            print(f"✓ Created view: {view_name}")
        except Exception as e:
            print(f"Error creating view {view_name}: {e}")
    
    # Bonus: Create a UNION view of ALL intraday data (super useful for cross-timeframe analysis)
    intraday_views = [f"bars_{i.replace('m','min')}" for i in INTERVALS if 'm' in i]
    if intraday_views:
        union_query = " UNION ALL ".join(f"SELECT * FROM {v}" for v in intraday_views)
        con.execute(f"""
        CREATE OR REPLACE VIEW all_intraday AS
        {union_query}
        """)
        print("✓ Created unified view: all_intraday")
    
    return con


# ── Example Queries ─────────────────────────────────────────────────────────────
def run_examples(con):
    print("\n=== Example Queries ===\n")
    
    # 1. How many bars per symbol in 1-minute data?
    print("Top 10 symbols by # of 1-min bars:")
    print(con.sql("""
        SELECT symbol, COUNT(*) AS bar_count
        FROM bars_1min
        GROUP BY symbol
        ORDER BY bar_count DESC
        LIMIT 10
    """).df())
    
    # 2. Average daily volume for high-volume stocks on a specific day
    print("\nHigh-volume stocks on 2026-01-15:")
    print(con.sql("""
        SELECT 
            symbol,
            SUM(volume) AS total_volume,
            AVG((close - open) / open * 100) AS avg_up_down_pct
        FROM bars_1min
        WHERE ts::DATE = '2026-01-15'
        GROUP BY symbol
        HAVING total_volume >= 5000000
        ORDER BY total_volume DESC
        LIMIT 20
    """).df())
    
    # 3. Up-bar % by timeframe for TSLA (your favorite example)
    print("\nTSLA up-bar % by timeframe:")
    print(con.sql("""
        SELECT 
            interval,
            COUNT(*) AS total_bars,
            SUM(CASE WHEN close > open THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS up_bar_pct
        FROM all_intraday
        WHERE symbol = 'TSLA'
        GROUP BY interval
        ORDER BY up_bar_pct DESC
    """).df())


# ── Main ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Option A: Persistent DB (recommended for repeated use)
    con = setup_qlab_duckdb()
    
    # Option B: In-memory only (faster first time, but gone when script ends)
    # con = duckdb.connect()
    # setup_qlab_duckdb(con=con)  # modify function if needed
    
    # Show all available views
    print("\nAvailable views:")
    print(con.sql("SHOW TABLES").df())
    
    # Run some example queries
    run_examples(con)
    
    # Interactive mode (optional - type SQL queries)
    print("\nType SQL queries below (or 'exit' to quit):")
    while True:
        q = input("> ").strip()
        if q.lower() in ('exit', 'q', 'quit'):
            break
        try:
            result = con.sql(q).df()
            print(result)
        except Exception as e:
            print(f"Error: {e}")