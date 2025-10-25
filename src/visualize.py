# Stdlib
from pathlib import Path

# Absolute Pfade relativ zur Projektwurzel (eine Ebene über src)
ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "plots"

# Third-party
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Projekt
from src.data import download_eth_1d
from src.features import add_features
from src.label import make_label
from src.model import train_logreg, infer_proba
from src.policy import ml_policy
from src.backtest import SimpleBacktester
from src.eval import returns_from_equity, sharpe, max_drawdown, cagr, sweep_threshold
from src.trades import compute_trades

def main():
    Path("plots").mkdir(parents=True, exist_ok=True)

    # 1) Daten + Features + Label
    df = download_eth_1d(start="2019-01-01")
    feat = add_features(df)
    lab  = make_label(feat, fee_buffer=0.0025)

    # 2) Split
    split_date = "2023-01-01"
    train = lab.loc[:split_date]
    test  = lab.loc[split_date:]

    # 3) Modell + Inferenz
    model = train_logreg(train)
    test_pred = infer_proba(model, test)
    
    # 4) Threshold-Sweep
    res = sweep_threshold(test_pred)
    best_thr = float(res.iloc[0]["p_thr"])

    # 5) Backtest mit best_thr
    signals = ml_policy(test_pred, p_thr=best_thr)
    bt = SimpleBacktester(test_pred)
    equity = bt.run(signals)
    ret = returns_from_equity(equity)

    # --- Trades berechnen und exportieren ---
    trades = compute_trades(test_pred, signals)
    Path("plots").mkdir(parents=True, exist_ok=True)
    trades.to_csv("plots/trades.csv", index=False)
    print("Trades:", len(trades))
    if len(trades) > 0:
        winrate = (trades["return"] > 0).mean()
        avg_ret = trades["return"].mean()
        med_ret = trades["return"].median()
        print(f"Winrate: {winrate:.2%} | Avg: {avg_ret:.3%} | Median: {med_ret:.3%}")
    else:
        print("Keine Trades exportiert.")

    # --- Ab hier wieder dein bisheriger Code ---
    # Ausgeführte Entries/Exits wie im Backtester (Signal am Vortag -> Aktion heute)
    d = test_pred.join(signals).dropna().copy()
    exec_entries_idx = []
    exec_exits_idx = []
    pos = 0  # 0: flat, 1: long

    for i in range(1, len(d)):
        prev = d.iloc[i-1]
        cur_idx = d.index[i]

        if pos == 0 and prev["entry_long"]:
            exec_entries_idx.append(cur_idx)
            pos = 1
        elif pos == 1 and prev["exit_long"]:
            exec_exits_idx.append(cur_idx)
            pos = 0

    # Kennzahlen
    s  = round(sharpe(ret, periods=252), 2)
    dd = round(max_drawdown(equity)*100, 2)
    cg = round(cagr(equity)*100, 2)

    print("\nThreshold-Sweep (Top 5):")
    print(res.head(5).to_string(index=False))
    print(f"\nBest p_thr: {best_thr}")
    print(f"Sharpe: {s}")
    print(f"CAGR%: {cg}")
    print(f"MaxDD%: {dd}")

    # --- Plot: Preis + EMAs + BUY/SELL-Marker (nur ausgeführte Trades!) ---
    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(x=feat.index, y=feat["Close"], name="Close", mode="lines"))
    price_fig.add_trace(go.Scatter(x=feat.index, y=feat["ema50"], name="EMA50", mode="lines"))
    price_fig.add_trace(go.Scatter(x=feat.index, y=feat["ema200"], name="EMA200", mode="lines"))

    entry_prices = feat.reindex(exec_entries_idx)["Close"]
    exit_prices  = feat.reindex(exec_exits_idx)["Close"]

    price_fig.add_trace(go.Scatter(
        x=exec_entries_idx, y=entry_prices, mode="markers",
        name="BUY", marker_symbol="triangle-up", marker_size=11
    ))
    price_fig.add_trace(go.Scatter(
        x=exec_exits_idx, y=exit_prices, mode="markers",
        name="SELL", marker_symbol="triangle-down", marker_size=11
    ))

    price_fig.update_layout(title="ETH Preis mit EMA50/EMA200", xaxis_title="Datum", yaxis_title="USD")
    price_fig.write_html("plots/price_ema.html")

    # 7) Plot: Equity-Kurve
    eq_fig = go.Figure()
    eq_fig.add_trace(go.Scatter(x=equity.index, y=equity, name="Equity", mode="lines"))
    eq_fig.update_layout(title=f"Equity-Kurve (p_thr={best_thr}, Sharpe={s}, MaxDD={dd}%)",
                         xaxis_title="Datum", yaxis_title="Equity")
    eq_fig.write_html("plots/equity_curve.html")

    # 8) Plot: Histogramm Returns
    hist_fig = px.histogram(pd.Series(ret, name="daily_ret"), x="daily_ret", nbins=60,
                            title="Histogramm täglicher Returns")
    hist_fig.write_html("plots/returns_hist.html")


if __name__ == "__main__":
    main()