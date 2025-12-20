"""
Vergleicht Performance mit/ohne Volumen-Features.
"""

from src.data import download_eth_1d
from src.features import add_features
from src.label import make_label
from src.config import FEATURES_BASE, FEATURES_WITH_VOLUME, P_ENTRY_THR, P_EXIT_THR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.policy import ml_policy
from src.backtest import SimpleBacktester
from src.eval import returns_from_equity, sharpe, max_drawdown, cagr
import pandas as pd


def train_and_test(train_df, test_df, features, feature_set_name):
    """Trainiert Modell und testet Performance."""
    # Training
    X_train = train_df[features].values
    y_train = train_df["y"].values
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=300))
    ])
    model.fit(X_train, y_train)

    # Inferenz
    X_test = test_df[features].values
    proba = model.predict_proba(X_test)[:, 1]
    test_pred = test_df.copy()
    test_pred["p_up"] = proba

    # Backtest
    signals_df = ml_policy(test_pred, p_entry_thr=P_ENTRY_THR, p_exit_thr=P_EXIT_THR)
    signals = signals_df[["entry_long", "exit_long"]].astype(int)

    bt = SimpleBacktester(test_pred)
    equity = bt.run(signals)
    ret = returns_from_equity(equity)

    return {
        "Feature_Set": feature_set_name,
        "Num_Features": len(features),
        "Entries": int(signals["entry_long"].sum()),
        "Sharpe": round(sharpe(ret, periods=252), 3),
        "CAGR%": round(cagr(equity) * 100, 2),
        "MaxDD%": round(max_drawdown(equity) * 100, 2),
        "Final_Equity": round(equity.iloc[-1], 3)
    }


def main():
    print("=" * 70)
    print("FEATURE COMPARISON - Mit/Ohne Volumen-Indikatoren")
    print("=" * 70)

    # 1) Daten laden
    print("\n[1/4] Lade Daten...")
    df = download_eth_1d(start="2019-01-01")

    # 2) Features MIT Volumen
    print("[2/4] Berechne Features mit Volumen...")
    feat_with_vol = add_features(df, include_volume=True)
    lab_with_vol = make_label(feat_with_vol, fee_buffer=0.0025)

    # 3) Features OHNE Volumen
    print("[3/4] Berechne Features ohne Volumen...")
    feat_no_vol = add_features(df, include_volume=False)
    lab_no_vol = make_label(feat_no_vol, fee_buffer=0.0025)

    # 4) Split
    split_date = "2023-01-01"
    train_with_vol = lab_with_vol.loc[:split_date]
    test_with_vol = lab_with_vol.loc[split_date:]
    train_no_vol = lab_no_vol.loc[:split_date]
    test_no_vol = lab_no_vol.loc[split_date:]

    # 5) Tests
    print("[4/4] Trainiere und teste Modelle...\n")
    results = []

    print("  -> Basis-Features (ohne Volumen)...")
    results.append(train_and_test(train_no_vol, test_no_vol, FEATURES_BASE, "Basis (ohne Volumen)"))

    print("  -> Mit Volumen-Features...")
    results.append(train_and_test(train_with_vol, test_with_vol, FEATURES_WITH_VOLUME, "Mit Volumen"))

    # Ergebnisse
    print("\n" + "=" * 70)
    print("ERGEBNISSE")
    print("=" * 70)
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    # Bestes
    best_idx = df_results["Sharpe"].idxmax()
    print("\n" + "=" * 70)
    print(f"BESTES FEATURE-SET: {df_results.iloc[best_idx]['Feature_Set']}")
    print("=" * 70)
    print(df_results.iloc[best_idx].to_string())
    print("=" * 70)


if __name__ == "__main__":
    main()
