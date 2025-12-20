"""
Vergleicht verschiedene ML-Modelle für den Trading Bot.

Testet:
1. Logistic Regression (Baseline)
2. Random Forest
3. Gradient Boosting (ähnlich XGBoost)
"""

from src.data import download_eth_1d
from src.features import add_features
from src.label import make_label
from src.model import train_logreg, train_random_forest, train_gradient_boosting, infer_proba
from src.policy import ml_policy
from src.backtest import SimpleBacktester
from src.eval import returns_from_equity, sharpe, max_drawdown, cagr
from src.config import P_ENTRY_THR, P_EXIT_THR
import pandas as pd


def test_model(model, model_name, test_pred):
    """Testet ein Modell und gibt Metriken zurück."""
    signals_df = ml_policy(
        test_pred,
        p_entry_thr=P_ENTRY_THR,
        p_exit_thr=P_EXIT_THR
    )
    signals = signals_df[["entry_long", "exit_long"]].astype(int)

    bt = SimpleBacktester(test_pred)
    equity = bt.run(signals)
    ret = returns_from_equity(equity)

    return {
        "Model": model_name,
        "Entries": int(signals["entry_long"].sum()),
        "Exits": int(signals["exit_long"].sum()),
        "Sharpe": round(sharpe(ret, periods=252), 3),
        "CAGR%": round(cagr(equity) * 100, 2),
        "MaxDD%": round(max_drawdown(equity) * 100, 2),
        "Final_Equity": round(equity.iloc[-1], 3)
    }


def main():
    print("=" * 70)
    print("MODEL COMPARISON - ETH Trading Bot")
    print("=" * 70)

    # 1) Daten laden
    print("\n[1/5] Lade Daten...")
    df = download_eth_1d(start="2019-01-01")
    feat = add_features(df)
    lab = make_label(feat, fee_buffer=0.0025)

    # 2) Train/Test-Split
    print("[2/5] Erstelle Train/Test-Split...")
    split_date = "2023-01-01"
    train = lab.loc[:split_date]
    test = lab.loc[split_date:]

    print(f"  Train: {train.index[0]} bis {train.index[-1]} ({len(train)} Tage)")
    print(f"  Test:  {test.index[0]} bis {test.index[-1]} ({len(test)} Tage)")

    results = []

    # 3) Logistic Regression (Baseline)
    print("\n[3/5] Trainiere Logistic Regression...")
    model_lr = train_logreg(train)
    test_pred_lr = infer_proba(model_lr, test)
    results.append(test_model(model_lr, "Logistic Regression", test_pred_lr))
    print("  -> Fertig")

    # 4) Random Forest
    print("\n[4/5] Trainiere Random Forest...")
    model_rf = train_random_forest(train, n_estimators=100, max_depth=10)
    test_pred_rf = infer_proba(model_rf, test)
    results.append(test_model(model_rf, "Random Forest", test_pred_rf))
    print("  -> Fertig")

    # 5) Gradient Boosting
    print("\n[5/5] Trainiere Gradient Boosting...")
    model_gb = train_gradient_boosting(train, n_estimators=100, max_depth=5, learning_rate=0.1)
    test_pred_gb = infer_proba(model_gb, test)
    results.append(test_model(model_gb, "Gradient Boosting", test_pred_gb))
    print("  -> Fertig")

    # Ergebnisse anzeigen
    print("\n" + "=" * 70)
    print("ERGEBNISSE")
    print("=" * 70)
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    # Bestes Modell highlighten
    best_idx = df_results["Sharpe"].idxmax()
    print("\n" + "=" * 70)
    print(f"BESTES MODELL (nach Sharpe): {df_results.iloc[best_idx]['Model']}")
    print("=" * 70)
    print(df_results.iloc[best_idx].to_string())
    print("=" * 70)


if __name__ == "__main__":
    main()
