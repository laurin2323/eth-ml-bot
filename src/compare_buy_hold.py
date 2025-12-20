"""
Vergleicht ML Trading Bot mit einfachem Buy & Hold.
"""

from src.data import download_eth_1d
from src.features import add_features
from src.label import make_label
from src.model import train_logreg, infer_proba
from src.policy import ml_policy
from src.backtest import SimpleBacktester
from src.eval import returns_from_equity, sharpe, max_drawdown, cagr
from src.config import P_ENTRY_THR, P_EXIT_THR
import pandas as pd


def main():
    print("=" * 70)
    print("BUY & HOLD vs ML TRADING BOT")
    print("=" * 70)

    # Daten laden
    print("\nLade Daten...")
    df = download_eth_1d(start="2019-01-01")
    feat = add_features(df)
    lab = make_label(feat, fee_buffer=0.0025)

    # Split
    split_date = "2023-01-01"
    train = lab.loc[:split_date]
    test = lab.loc[split_date:]

    print(f"Test-Period: {test.index[0]} bis {test.index[-1]} ({len(test)} Tage)")
    print(f"Start-Preis: ${test.iloc[0]['Close']:.2f}")
    print(f"End-Preis: ${test.iloc[-1]['Close']:.2f}")

    # 1) Buy & Hold
    print("\n" + "-" * 70)
    print("STRATEGIE 1: BUY & HOLD")
    print("-" * 70)

    start_price = test.iloc[0]["Close"]
    end_price = test.iloc[-1]["Close"]

    # Fees beim Kauf (einmalig)
    buy_price = start_price * 1.0025  # 0.25% fees

    bh_return = (end_price / buy_price - 1) * 100
    bh_equity = end_price / buy_price

    # Berechne CAGR und MaxDD für Buy & Hold
    bh_equity_series = test["Close"] / buy_price
    bh_ret = returns_from_equity(bh_equity_series)
    bh_sharpe = sharpe(bh_ret, periods=252)
    bh_cagr = cagr(bh_equity_series) * 100
    bh_maxdd = max_drawdown(bh_equity_series) * 100

    print(f"Return: {bh_return:.2f}%")
    print(f"Sharpe: {bh_sharpe:.3f}")
    print(f"CAGR: {bh_cagr:.2f}%")
    print(f"Max Drawdown: {bh_maxdd:.2f}%")
    print(f"Final Equity: {bh_equity:.3f}")

    # 2) ML Trading Bot
    print("\n" + "-" * 70)
    print("STRATEGIE 2: ML TRADING BOT")
    print("-" * 70)

    model = train_logreg(train)
    test_pred = infer_proba(model, test)
    signals_df = ml_policy(test_pred, p_entry_thr=P_ENTRY_THR, p_exit_thr=P_EXIT_THR)
    signals = signals_df[["entry_long", "exit_long"]].astype(int)

    bt = SimpleBacktester(test_pred)
    ml_equity = bt.run(signals)
    ml_ret = returns_from_equity(ml_equity)

    ml_return = (ml_equity.iloc[-1] - 1) * 100
    ml_sharpe = sharpe(ml_ret, periods=252)
    ml_cagr = cagr(ml_equity) * 100
    ml_maxdd = max_drawdown(ml_equity) * 100
    n_trades = int(signals["entry_long"].sum())

    print(f"Return: {ml_return:.2f}%")
    print(f"Sharpe: {ml_sharpe:.3f}")
    print(f"CAGR: {ml_cagr:.2f}%")
    print(f"Max Drawdown: {ml_maxdd:.2f}%")
    print(f"Final Equity: {ml_equity.iloc[-1]:.3f}")
    print(f"Number of Trades: {n_trades}")

    # Vergleich
    print("\n" + "=" * 70)
    print("VERGLEICH")
    print("=" * 70)

    results = pd.DataFrame({
        "Metrik": ["Return %", "Sharpe", "CAGR %", "MaxDD %", "Final Equity", "Trades"],
        "Buy & Hold": [
            f"{bh_return:.2f}",
            f"{bh_sharpe:.3f}",
            f"{bh_cagr:.2f}",
            f"{bh_maxdd:.2f}",
            f"{bh_equity:.3f}",
            "1"
        ],
        "ML Bot": [
            f"{ml_return:.2f}",
            f"{ml_sharpe:.3f}",
            f"{ml_cagr:.2f}",
            f"{ml_maxdd:.2f}",
            f"{ml_equity.iloc[-1]:.3f}",
            f"{n_trades}"
        ],
        "Differenz": [
            f"{ml_return - bh_return:+.2f}",
            f"{ml_sharpe - bh_sharpe:+.3f}",
            f"{ml_cagr - bh_cagr:+.2f}",
            f"{ml_maxdd - bh_maxdd:+.2f}",
            f"{ml_equity.iloc[-1] - bh_equity:+.3f}",
            f"{n_trades - 1:+d}"
        ]
    })

    print(results.to_string(index=False))

    if ml_return > bh_return:
        print("\n✅ ML Bot schlägt Buy & Hold!")
    else:
        print("\n❌ Buy & Hold schlägt ML Bot")
        print(f"\nUnderperformance: {bh_return - ml_return:.2f}%")


if __name__ == "__main__":
    main()
