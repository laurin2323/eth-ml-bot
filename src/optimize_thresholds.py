"""
Optimiert Entry- und Exit-Thresholds auf Validation-Set.
"""

from src.data import download_eth_1d
from src.features import add_features
from src.label import make_label
from src.model import train_logreg, infer_proba
from src.eval import sweep_threshold
import numpy as np


def main():
    print("=" * 70)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 70)

    # 1) Daten
    print("\n[1/4] Lade Daten...")
    df = download_eth_1d(start="2019-01-01")
    feat = add_features(df, include_volume=False)  # Basis-Features
    lab = make_label(feat, fee_buffer=0.0025)

    # 2) Split: Train / Validation / Test
    print("[2/4] Erstelle Train/Validation/Test-Split...")
    split_date = "2023-01-01"
    val_date = "2022-06-01"

    train_full = lab.loc[:split_date]
    test = lab.loc[split_date:]

    train = train_full.loc[:val_date]
    val = train_full.loc[val_date:]

    print(f"  Train:      {train.index[0]} bis {train.index[-1]} ({len(train)} Tage)")
    print(f"  Validation: {val.index[0]} bis {val.index[-1]} ({len(val)} Tage)")
    print(f"  Test:       {test.index[0]} bis {test.index[-1]} ({len(test)} Tage)")

    # 3) Modell auf TRAIN trainieren
    print("\n[3/4] Trainiere Modell auf Train-Set...")
    model = train_logreg(train)
    print("  -> Fertig")

    # 4) Threshold-Sweep auf VALIDATION
    print("\n[4/4] Optimiere Thresholds auf Validation-Set...")
    val_pred = infer_proba(model, val)

    # Grid für Entry-Thresholds
    entry_grid = np.linspace(0.30, 0.70, 21)  # 0.30 bis 0.70 in 21 Schritten

    # Grid für Exit-Thresholds
    exit_grid = [0.2, 0.3, 0.4]

    best_sharpe = -999
    best_config = None
    all_results = []

    for p_exit in exit_grid:
        print(f"\n  Testing p_exit_thr = {p_exit}...")
        res = sweep_threshold(val_pred, entry_thresholds=entry_grid, p_exit_thr=p_exit)

        # Bestes für diesen Exit-Threshold
        best_row = res.iloc[0]
        all_results.append(best_row)

        if best_row["sharpe"] > best_sharpe:
            best_sharpe = best_row["sharpe"]
            best_config = best_row

    # Ergebnisse
    print("\n" + "=" * 70)
    print("TOP KONFIGURATIONEN (eine pro Exit-Threshold)")
    print("=" * 70)
    import pandas as pd
    df_results = pd.DataFrame(all_results)
    print(df_results.to_string(index=False))

    print("\n" + "=" * 70)
    print("BESTE KONFIGURATION (Validation-Set)")
    print("=" * 70)
    print(f"p_entry_thr: {best_config['p_entry_thr']}")
    print(f"p_exit_thr:  {best_config['p_exit_thr']}")
    print(f"Sharpe:      {best_config['sharpe']:.3f}")
    print(f"CAGR:        {best_config['cagr']*100:.2f}%")
    print(f"MaxDD:       {best_config['maxdd']*100:.2f}%")
    print("=" * 70)

    print("\nHINWEIS: Setze diese Werte in src/config.py ein:")
    print(f"  P_ENTRY_THR = {best_config['p_entry_thr']}")
    print(f"  P_EXIT_THR  = {best_config['p_exit_thr']}")


if __name__ == "__main__":
    main()
