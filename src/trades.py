# src/trades.py
from pathlib import Path
import pandas as pd

def compute_trades(df_with_proba: pd.DataFrame, signals: pd.DataFrame, fees_bps: int = 20, slippage_bps: int = 5) -> pd.DataFrame:
    """Erzeuge saubere Entry/Exit-Paare (1 Position max) mit Fees und Slippage."""
    d = df_with_proba.join(signals).dropna().copy()

    fee = fees_bps / 10000   # z. B. 20 bps = 0.2 %
    slip = slippage_bps / 10000

    trades = []
    pos = 0
    entry_idx = None
    entry_price = None

    for i in range(1, len(d)):
        prev = d.iloc[i-1]
        cur = d.iloc[i]
        cur_idx = d.index[i]

        if pos == 0 and prev["entry_long"]:
            pos = 1
            entry_idx = cur_idx
            entry_price = cur["Open"] * (1 + slip + fee)

        elif pos == 1 and prev["exit_long"]:
            exit_idx = cur_idx
            exit_price = cur["Open"] * (1 - slip - fee)
            ret = exit_price / entry_price - 1
            trades.append({
                "entry_date": entry_idx, "entry_price": entry_price,
                "exit_date": exit_idx,   "exit_price": exit_price,
                "return": ret
            })
            pos = 0
            entry_idx = None
            entry_price = None

    return pd.DataFrame(trades)

