# src/tune.py
import numpy as np
import pandas as pd
from .policy import ml_policy
from .backtest import SimpleBacktester
from .eval import returns_from_equity, sharpe, max_drawdown, cagr

def sweep_threshold(df_with_proba, thr_list=None):
    if thr_list is None:
        thr_list = np.round(np.linspace(0.50, 0.70, 9), 2)
    rows = []
    for thr in thr_list:
        signals = ml_policy(df_with_proba, p_thr=float(thr))
        bt = SimpleBacktester(df_with_proba)
        equity = bt.run(signals)
        ret = returns_from_equity(equity)
        rows.append({
            "p_thr": float(thr),
            "Sharpe": round(sharpe(ret), 3),
            "MaxDD%": round(max_drawdown(equity)*100, 2),
            "CAGR%": round(cagr(equity)*100, 2)
        })
    return pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
