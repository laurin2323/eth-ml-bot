import numpy as np
import pandas as pd

def returns_from_equity(equity: pd.Series):
    return equity.pct_change().fillna(0.0)

def sharpe(returns, periods=252, rf=0.0):
    r = np.array(returns)
    if r.std() == 0: return 0.0
    return (r.mean() - rf) / r.std() * np.sqrt(periods)

def max_drawdown(equity: pd.Series):
    cummax = equity.cummax()
    dd = equity / cummax - 1
    return dd.min()
