import pandas as pd
from pathlib import Path

ticker = "AAPL"
bs_path = Path("market_data/yfinance/fundamentals/balance_sheet") / f"{ticker}.parquet"

if bs_path.exists():
    df = pd.read_parquet(bs_path)
    print(f"Shape: {df.shape}")
    print("Index type:", df.index.dtype)
    print("Index range:", df.index.min(), "→", df.index.max())
    print("Columns:", list(df.columns))
    print("\nSample rows (head + tail):")
    print(df.head(8))
    print("...")
    print(df.tail(8))
    
    # Look for total assets variants
    asset_cols = [c for c in df.columns if "asset" in c.lower() or "total" in c.lower()]
    print("\nPotential asset columns:", asset_cols)
else:
    print(f"File missing: {bs_path}")