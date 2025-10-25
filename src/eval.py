import numpy as np
import pandas as pd

def returns_from_equity(equity: pd.Series):
    return equity.pct_change().fillna(0.0)

def sharpe(returns, periods=252, rf=0.0):
    r = np.array(returns)
    if r.std() == 0:
        return 0.0
    return (r.mean() - rf) / r.std() * np.sqrt(periods)

def max_drawdown(equity: pd.Series):
    cummax = equity.cummax()
    dd = equity / cummax - 1
    return dd.min()

def cagr(equity: pd.Series, periods_per_year=252):
    # Compound Annual Growth Rate
    if len(equity) < 2:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0]
    years = len(equity) / periods_per_year
    if years <= 0:
        return 0.0
    return total_return**(1/years) - 1

def sweep_threshold(df, thresholds=None):
    import numpy as np
    import pandas as pd
    from src.policy import ml_policy
    from src.backtest import SimpleBacktester
    from src.eval import returns_from_equity, sharpe

    if thresholds is None:
        thresholds = np.linspace(0.4, 0.6, 21)

    results = []
    for t in thresholds:
        signals = ml_policy(df, p_thr=t)
        bt = SimpleBacktester(df)
        equity = bt.run(signals)
        ret = returns_from_equity(equity)
        s = sharpe(ret, periods=252)
        results.append({"p_thr": t, "sharpe": s})

    res = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    return res
