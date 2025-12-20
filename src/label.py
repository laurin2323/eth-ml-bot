"""
Modul zur Label-Generierung für ML-Training.
"""

import pandas as pd

def make_label(df: pd.DataFrame, fee_buffer=0.0025, forward_days=1) -> pd.DataFrame:
    """
    Erstellt binäre Labels basierend auf Forward-Returns.

    Label y=1 wenn der Forward-Return größer als fee_buffer ist,
    sonst y=0. Der fee_buffer berücksichtigt Trading-Kosten.

    Args:
        df: DataFrame mit OHLCV-Daten und Features
        fee_buffer: Mindest-Return um profitable zu sein (default: 0.0025 = 0.25%)
        forward_days: Anzahl Tage für Forward-Return (default: 1)
                      - 1 = Next-day return (noise)
                      - 5 = 5-day return (besseres Signal für ML)

    Returns:
        DataFrame mit zusätzlichen Spalten:
        - ret_fwd: Forward-Return (forward_days ahead)
        - y: Binäres Label (1 = profitabel, 0 = nicht profitabel)
    """
    d = df.copy()
    d["ret_fwd"] = d["Close"].shift(-forward_days)/d["Close"] - 1
    # Adjust fee_buffer for multi-day: bei 5 Tagen erwarten wir min. 5x die fees
    adjusted_buffer = fee_buffer * forward_days
    d["y"] = (d["ret_fwd"] > adjusted_buffer).astype(int)
    return d.dropna()
