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

def sweep_threshold(df, entry_thresholds=None, p_exit_thr=0.4):
    """
    Optimiert Entry-Threshold auf Validation-Set.

    Args:
        df: DataFrame mit Preis-Daten und p_up-Prognose
        entry_thresholds: Liste von Entry-Thresholds zum Testen (default: 0.4-0.6)
        p_exit_thr: Fixer Exit-Threshold (default: 0.4)

    Returns:
        DataFrame mit Sharpe/CAGR/MaxDD für jeden Threshold, sortiert nach Sharpe (absteigend)
    """
    import numpy as np
    import pandas as pd
    from src.policy import ml_policy
    from src.backtest import SimpleBacktester
    from src.eval import returns_from_equity, sharpe, max_drawdown, cagr

    if entry_thresholds is None:
        entry_thresholds = np.linspace(0.4, 0.6, 21)

    results = []
    for t in entry_thresholds:
        signals = ml_policy(df, p_entry_thr=float(t), p_exit_thr=float(p_exit_thr))
        bt = SimpleBacktester(df)
        equity = bt.run(signals)
        ret = returns_from_equity(equity)

        n_entries = int(signals["entry_long"].sum())
        if n_entries < 5:        # Mindestanzahl Trades
            s = -1e9             # harte Strafe für "macht nichts"
        else:
            s = sharpe(ret, periods=252)

        results.append({
            "p_entry_thr": float(t),
            "p_exit_thr": float(p_exit_thr),
            "sharpe": float(s),
            "maxdd": float(max_drawdown(equity)),
            "cagr": float(cagr(equity))
        })

    return (
        pd.DataFrame(results)
        .sort_values("sharpe", ascending=False)
        .reset_index(drop=True)
    )

