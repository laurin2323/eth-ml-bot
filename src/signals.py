# src/signals.py
from src.data import download_eth_1d
from src.features import add_features
from src.label import make_label
from src.model import train_logreg, infer_proba
from src.policy import ml_policy
from pathlib import Path
import pandas as pd

def main():
    Path("plots").mkdir(parents=True, exist_ok=True)

    df = download_eth_1d(start="2019-01-01")
    feat = add_features(df)
    lab  = make_label(feat, fee_buffer=0.0025)

    split_date = "2023-01-01"
    train = lab.loc[:split_date]
    test  = lab.loc[split_date:]

    model = train_logreg(train)
    test_pred = infer_proba(model, test)

    # Threshold kannst du später tunen; nimm vorerst 0.55
    signals = ml_policy(test_pred, p_thr=0.55)

    out = test_pred.join(signals).dropna().copy()
    # Nur tatsächliche Entry-Zeilen
    entries = out[out["entry_long"] == True].copy()
    entries = entries[["Close","p_up","rsi14","atr_pct","ema50","ema200"]]
    entries = entries.rename(columns={"Close":"price"})
    entries.index.name = "date"

    # CSV speichern
    entries.to_csv("plots/entries.csv")
    print(f"{len(entries)} Kaufsignale exportiert -> plots/entries.csv")
    print(entries.head(10).to_string())

if __name__ == "__main__":
    main()
