"""
Vergleicht verbesserte ML Trading Bots mit Buy & Hold.

Testet:
1. Buy & Hold (Baseline)
2. ML Bot Long-Only (Original)
3. ML Bot Long/Short + 5-Day Labels (IMPROVED)
"""

from src.data import download_eth_1d
from src.features import add_features
from src.label import make_label
from src.model import train_logreg, infer_proba
from src.policy import ml_policy, ml_policy_longshort
from src.backtest import SimpleBacktester, LongShortBacktester
from src.eval import returns_from_equity, sharpe, max_drawdown, cagr
from src.config import P_ENTRY_THR, P_EXIT_THR
import pandas as pd


def main():
    print("=" * 70)
    print("IMPROVED ML BOT vs ORIGINAL vs BUY & HOLD")
    print("=" * 70)

    # Daten laden
    print("\nLade Daten...")
    df = download_eth_1d(start="2019-01-01")
    feat = add_features(df)

    # Split
    split_date = "2023-01-01"
    train_full = feat.loc[:split_date]
    test = feat.loc[split_date:]

    print(f"Test-Period: {test.index[0]} bis {test.index[-1]} ({len(test)} Tage)")
    print(f"Start-Preis: ${test.iloc[0]['Close']:.2f}")
    print(f"End-Preis: ${test.iloc[-1]['Close']:.2f}")

    # ========================================================================
    # 1) Buy & Hold
    # ========================================================================
    print("\n" + "-" * 70)
    print("STRATEGIE 1: BUY & HOLD")
    print("-" * 70)

    start_price = test.iloc[0]["Close"]
    end_price = test.iloc[-1]["Close"]
    buy_price = start_price * 1.0025

    bh_equity_series = test["Close"] / buy_price
    bh_ret = returns_from_equity(bh_equity_series)
    bh_return = (end_price / buy_price - 1) * 100
    bh_sharpe = sharpe(bh_ret, periods=252)
    bh_cagr = cagr(bh_equity_series) * 100
    bh_maxdd = max_drawdown(bh_equity_series) * 100

    print(f"Return: {bh_return:.2f}%")
    print(f"Sharpe: {bh_sharpe:.3f}")
    print(f"CAGR: {bh_cagr:.2f}%")
    print(f"Max Drawdown: {bh_maxdd:.2f}%")
    print(f"Final Equity: {bh_equity_series.iloc[-1]:.3f}")

    # ========================================================================
    # 2) ML Bot Long-Only (Original)
    # ========================================================================
    print("\n" + "-" * 70)
    print("STRATEGIE 2: ML BOT LONG-ONLY (Original)")
    print("-" * 70)

    # 1-Day Labels
    lab_1d = make_label(train_full, fee_buffer=0.0025, forward_days=1)
    model_1d = train_logreg(lab_1d)

    test_1d = make_label(test, fee_buffer=0.0025, forward_days=1)
    test_pred_1d = infer_proba(model_1d, test_1d)
    signals_longonly = ml_policy(test_pred_1d, p_entry_thr=P_ENTRY_THR, p_exit_thr=P_EXIT_THR)

    bt_longonly = SimpleBacktester(test_pred_1d)
    equity_longonly = bt_longonly.run(signals_longonly[["entry_long", "exit_long"]].astype(int))
    ret_longonly = returns_from_equity(equity_longonly)

    longonly_return = (equity_longonly.iloc[-1] - 1) * 100
    longonly_sharpe = sharpe(ret_longonly, periods=252)
    longonly_cagr = cagr(equity_longonly) * 100
    longonly_maxdd = max_drawdown(equity_longonly) * 100
    longonly_trades = int(signals_longonly["entry_long"].sum())

    print(f"Return: {longonly_return:.2f}%")
    print(f"Sharpe: {longonly_sharpe:.3f}")
    print(f"CAGR: {longonly_cagr:.2f}%")
    print(f"Max Drawdown: {longonly_maxdd:.2f}%")
    print(f"Final Equity: {equity_longonly.iloc[-1]:.3f}")
    print(f"Trades: {longonly_trades}")

    # ========================================================================
    # 3) ML Bot Long/Short + 5-Day Labels (IMPROVED)
    # ========================================================================
    print("\n" + "-" * 70)
    print("STRATEGIE 3: ML BOT LONG/SHORT + 5-DAY LABELS (IMPROVED)")
    print("-" * 70)

    # 5-Day Labels
    lab_5d = make_label(train_full, fee_buffer=0.0025, forward_days=5)
    model_5d = train_logreg(lab_5d)

    test_5d = make_label(test, fee_buffer=0.0025, forward_days=5)
    test_pred_5d = infer_proba(model_5d, test_5d)

    # Long/Short Policy (ohne Filter = aggressiv)
    signals_longshort = ml_policy_longshort(
        test_pred_5d,
        p_long_thr=0.55,
        p_short_thr=0.45,
        use_filters=False
    )

    bt_longshort = LongShortBacktester(test_pred_5d)
    equity_longshort = bt_longshort.run(signals_longshort)
    ret_longshort = returns_from_equity(equity_longshort)

    longshort_return = (equity_longshort.iloc[-1] - 1) * 100
    longshort_sharpe = sharpe(ret_longshort, periods=252)
    longshort_cagr = cagr(equity_longshort) * 100
    longshort_maxdd = max_drawdown(equity_longshort) * 100
    n_longs = int(signals_longshort["entry_long"].sum())
    n_shorts = int(signals_longshort["entry_short"].sum())

    print(f"Return: {longshort_return:.2f}%")
    print(f"Sharpe: {longshort_sharpe:.3f}")
    print(f"CAGR: {longshort_cagr:.2f}%")
    print(f"Max Drawdown: {longshort_maxdd:.2f}%")
    print(f"Final Equity: {equity_longshort.iloc[-1]:.3f}")
    print(f"Long Trades: {n_longs}")
    print(f"Short Trades: {n_shorts}")
    print(f"Total Trades: {n_longs + n_shorts}")

    # ========================================================================
    # VERGLEICH
    # ========================================================================
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
            f"{bh_equity_series.iloc[-1]:.3f}",
            "1"
        ],
        "ML Long-Only": [
            f"{longonly_return:.2f}",
            f"{longonly_sharpe:.3f}",
            f"{longonly_cagr:.2f}",
            f"{longonly_maxdd:.2f}",
            f"{equity_longonly.iloc[-1]:.3f}",
            f"{longonly_trades}"
        ],
        "ML Long/Short (5d)": [
            f"{longshort_return:.2f}",
            f"{longshort_sharpe:.3f}",
            f"{longshort_cagr:.2f}",
            f"{longshort_maxdd:.2f}",
            f"{equity_longshort.iloc[-1]:.3f}",
            f"{n_longs + n_shorts}"
        ]
    })

    print(results.to_string(index=False))

    # Analyse
    print("\n" + "=" * 70)
    print("ANALYSE")
    print("=" * 70)

    if longshort_return > bh_return:
        print("-> IMPROVED Bot schlaegt Buy & Hold!")
        print(f"   Outperformance: +{longshort_return - bh_return:.2f}%")
    elif longshort_return > longonly_return:
        print("-> IMPROVED Bot schlaegt Original Long-Only!")
        print(f"   Verbesserung: +{longshort_return - longonly_return:.2f}%")
        print(f"   Aber immer noch unter Buy & Hold: {bh_return - longshort_return:.2f}%")
    else:
        print("-> Verbesserungen haben nicht geholfen")
        print(f"   Long-Only bleibt besser")

    print("\nVERBESSERUNGEN:")
    print(f"1. 5-Day Labels: Weniger Noise im Training")
    print(f"2. Long/Short: 100% Market Exposure (vs {longonly_trades}/{len(test)}={longonly_trades/len(test)*100:.1f}%)")
    print(f"3. Relaxed Rules: Mehr Trading-Opportunities")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
