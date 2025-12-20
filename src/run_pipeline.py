from src.data import download_eth_1d
from src.features import add_features
from src.label import make_label
from src.model import train_logreg, infer_proba
from src.policy import ml_policy
from src.backtest import SimpleBacktester
from src.eval import returns_from_equity, sharpe, max_drawdown, cagr
from src.config import P_ENTRY_THR, P_EXIT_THR

def main():
    """
    Einfaches Pipeline-Script für schnelles Testen.

    Flow:
    1. Daten laden
    2. Features berechnen
    3. Labels generieren
    4. Train/Test-Split
    5. Modell trainieren
    6. Signale generieren
    7. Backtest durchführen
    8. Metriken ausgeben
    """
    # 1) Daten
    df = download_eth_1d(start="2019-01-01")

    # 2) Features
    feat = add_features(df)

    # 3) Label
    lab = make_label(feat, fee_buffer=0.0025)

    # 4) Zeitbasierter Split
    split_date = "2023-01-01"
    train = lab.loc[:split_date]
    test  = lab.loc[split_date:]

    # 5) Modell
    model = train_logreg(train)

    # 6) Inferenz + Policy
    test_pred = infer_proba(model, test)

    signals_df = ml_policy(
        test_pred,
        p_entry_thr=P_ENTRY_THR,
        p_exit_thr=P_EXIT_THR
    )

    signals = signals_df[["entry_long", "exit_long"]].astype(int)

    print("=" * 50)
    print("KONFIGURATION")
    print("=" * 50)
    print(f"P_ENTRY_THR: {P_ENTRY_THR} | P_EXIT_THR: {P_EXIT_THR}")
    print(f"Entry-Signale: {int(signals['entry_long'].sum())}")
    print(f"Exit-Signale: {int(signals['exit_long'].sum())}")

    # 7) Backtest
    bt = SimpleBacktester(test_pred)
    equity = bt.run(signals)

    # 8) Kennzahlen
    ret = returns_from_equity(equity)
    print("\n" + "=" * 50)
    print("BACKTEST-ERGEBNISSE")
    print("=" * 50)
    print(f"Sharpe Ratio: {round(sharpe(ret, periods=252), 2)}")
    print(f"CAGR: {round(cagr(equity) * 100, 2)}%")
    print(f"Max Drawdown: {round(max_drawdown(equity) * 100, 1)}%")
    print(f"Final Equity: {round(equity.iloc[-1], 2)}")
    print("=" * 50)

if __name__ == "__main__":
    main()
