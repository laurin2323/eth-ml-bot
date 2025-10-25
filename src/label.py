import pandas as pd

def make_label(df: pd.DataFrame, fee_buffer=0.0025) -> pd.DataFrame:
    d = df.copy()
    d["ret_fwd"] = d["Close"].shift(-1)/d["Close"] - 1
    d["y"] = (d["ret_fwd"] > fee_buffer).astype(int)
    return d.dropna()
