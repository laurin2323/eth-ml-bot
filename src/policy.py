import pandas as pd

def ml_policy(df: pd.DataFrame, p_thr=0.55):
    entry = (df["p_up"] > p_thr) & (df["atr_pct"].between(0.8, 6.0)) & (df["Close"] > df["ema50"])
    exit_  = (df["rsi14"] > 55) | (df["p_up"] < 0.50)
    return pd.DataFrame({"entry_long": entry, "exit_long": exit_}, index=df.index)
