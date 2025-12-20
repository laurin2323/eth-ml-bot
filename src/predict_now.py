"""
Live Trading Signal Generator - gibt aktuelle Kauf/Verkauf/Halten-Empfehlung.

Verwendung:
    python -m src.predict_now
"""

from src.data import download_eth_1d
from src.features import add_features
from src.label import make_label
from src.model import train_logreg, infer_proba
from src.policy import ml_policy
from src.config import P_ENTRY_THR, P_EXIT_THR
import pandas as pd
from datetime import datetime

def main():
    print("=" * 70)
    print("ETH TRADING BOT - LIVE PREDICTION")
    print("=" * 70)
    print(f"Zeitpunkt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. Lade aktuelle Daten (inkl. heute)
    print("Lade aktuelle Daten...")
    df = download_eth_1d(start="2019-01-01")  # end=None -> bis heute
    feat = add_features(df)
    lab = make_label(feat, fee_buffer=0.0025)

    # 2. Trainiere Modell auf ALLEN Daten bis gestern
    print("Trainiere Modell auf allen verfügbaren Daten...")
    split_date = "2023-01-01"  # Train/Val split für Modell-Training
    train = lab.loc[:split_date]

    model = train_logreg(train)

    # 3. Prognose für HEUTE (letzte verfügbare Zeile)
    latest = lab.iloc[[-1]]  # Letzte Zeile als DataFrame
    latest_pred = infer_proba(model, latest)

    # 4. Policy anwenden
    signals_df = ml_policy(latest_pred, p_entry_thr=P_ENTRY_THR, p_exit_thr=P_EXIT_THR)

    # 5. Extrahiere Informationen
    latest_date = latest.index[0]
    latest_price = latest["Close"].iloc[0]
    p_up = latest_pred["p_up"].iloc[0]
    entry_signal = signals_df["entry_long"].iloc[0]
    exit_signal = signals_df["exit_long"].iloc[0]

    # Technische Indikatoren
    rsi = latest["rsi14"].iloc[0]
    ema50 = latest["ema50"].iloc[0]
    atr_pct = latest["atr_pct"].iloc[0]

    # 6. Ausgabe
    print("\n" + "-" * 70)
    print("MARKTDATEN")
    print("-" * 70)
    print(f"Datum: {latest_date.strftime('%Y-%m-%d')}")
    print(f"ETH Preis: ${latest_price:,.2f}")
    print(f"RSI(14): {rsi:.1f}")
    print(f"EMA(50): ${ema50:,.2f}")
    print(f"ATR: {atr_pct:.2f}%")

    print("\n" + "-" * 70)
    print("ML PROGNOSE")
    print("-" * 70)
    print(f"Wahrscheinlichkeit UP: {p_up:.1%}")
    print(f"Entry Threshold: {P_ENTRY_THR:.1%}")
    print(f"Exit Threshold: {P_EXIT_THR:.1%}")

    # 7. Trading Empfehlung
    print("\n" + "=" * 70)
    print("TRADING SIGNAL")
    print("=" * 70)

    if entry_signal:
        print("SIGNAL: BUY")
        print("Empfehlung: Long Position eröffnen")
        print(f"Begründung: ML-Prognose {p_up:.1%} > Entry-Threshold {P_ENTRY_THR:.1%}")
        print(f"             + ATR in Range ({atr_pct:.2f}% zwischen 0.8-6.0%)")
        print(f"             + Preis über EMA50 (${latest_price:.2f} > ${ema50:.2f})")
    elif exit_signal:
        print("SIGNAL: SELL")
        print("Empfehlung: Bestehende Long Position schliessen")
        if rsi > 55:
            print(f"Begründung: RSI überkauft ({rsi:.1f} > 55)")
        if p_up < P_EXIT_THR:
            print(f"Begründung: ML-Prognose gesunken ({p_up:.1%} < {P_EXIT_THR:.1%})")
    else:
        print("SIGNAL: HOLD")
        print("Empfehlung: Abwarten, keine Action")
        print(f"Begründung: ML-Prognose {p_up:.1%} nicht stark genug")
        if atr_pct < 0.8 or atr_pct > 6.0:
            print(f"             + ATR ausserhalb Range ({atr_pct:.2f}%)")
        if latest_price <= ema50:
            print(f"             + Preis unter EMA50 (${latest_price:.2f} <= ${ema50:.2f})")

    print("=" * 70)

    # 8. Zusätzliche Info
    print("\nHINWEIS:")
    print("- Dies ist eine ML-basierte Prognose, keine Anlageberatung")
    print("- Berücksichtige immer dein eigenes Risikomanagement")
    print("- Backtesting-Performance: Siehe compare_buy_hold.py")
    print("\nUm täglich zu aktualisieren, führe erneut aus:")
    print("  python -m src.predict_now")

if __name__ == "__main__":
    main()
