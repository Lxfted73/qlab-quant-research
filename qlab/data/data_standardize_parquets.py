# standardize_parquets.py
from pathlib import Path
import pandas as pd

folder = Path("data/yfinance/1m")
for f in folder.glob("*.parquet"):
    try:
        df = pd.read_parquet(f)
        
        # Force datetime index named "Datetime"
        if not isinstance(df.index, pd.DatetimeIndex):
            # guess which column is time
            time_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
            if time_cols:
                df = df.set_index(time_cols[0])
            else:
                print(f"Skipping {f.name} - no obvious datetime column")
                continue
        
        df.index.name = "Datetime"
        df = df.sort_index()
        
        # Optional: drop any unnamed index columns
        if "__index_level_0__" in df.columns:
            df = df.drop(columns="__index_level_0__")
            
        df.to_parquet(f, index=True)
        print(f"Fixed: {f.name}")
    except Exception as e:
        print(f"Error on {f.name}: {e}")