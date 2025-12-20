"""
Modul für Trading-Policy (Entry/Exit-Regeln).
"""

import pandas as pd

def ml_policy(
    df: pd.DataFrame,
    p_entry_thr=0.6,
    p_exit_thr=0.4
):
    """
    Generiert Entry/Exit-Signale basierend auf ML-Prognosen und technischen Filtern.

    Entry-Bedingungen (alle müssen erfüllt sein):
    - ML-Wahrscheinlichkeit für "up" > p_entry_thr
    - ATR zwischen 0.8% und 6.0% (Volatilitätsfilter)
    - Preis über EMA50 (Trendfilter)

    Exit-Bedingungen (mindestens eine):
    - RSI über 55 (überkauft)
    - ML-Wahrscheinlichkeit unter p_exit_thr

    Args:
        df: DataFrame mit Features und ML-Prognose 'p_up'
        p_entry_thr: Threshold für Entry-Signal (default: 0.6)
        p_exit_thr: Threshold für Exit-Signal (default: 0.4)

    Returns:
        DataFrame mit Boolean-Spalten:
        - entry_long: True wenn Entry-Bedingungen erfüllt
        - exit_long: True wenn Exit-Bedingungen erfüllt
    """
    entry = (
        (df["p_up"] > p_entry_thr)
        & (df["atr_pct"].between(0.8, 6.0))
        & (df["Close"] > df["ema50"])
    )
    exit_ = (
        (df["rsi14"] > 55)
        | (df["p_up"] < p_exit_thr)
    )
    return pd.DataFrame({"entry_long": entry, "exit_long": exit_}, index=df.index)


def ml_policy_longshort(
    df: pd.DataFrame,
    p_long_thr=0.55,
    p_short_thr=0.45,
    use_filters=False
):
    """
    Long/Short Policy - immer investiert (entweder Long ODER Short).

    Diese Policy verbessert die Market Exposure dramatisch:
    - Long-only: ~2.6% exposure (28 Trades in 1084 Tagen)
    - Long/Short: 100% exposure (immer im Markt)

    Signale:
    - entry_long: p_up > p_long_thr (Bullish)
    - entry_short: p_up < p_short_thr (Bearish)
    - Zwischen Thresholds: Halte aktuelle Position

    Args:
        df: DataFrame mit Features und ML-Prognose 'p_up'
        p_long_thr: Threshold für Long-Entry (default: 0.55)
        p_short_thr: Threshold für Short-Entry (default: 0.45)
        use_filters: Wenn True, verwende ATR/EMA Filter (default: False = relaxed)

    Returns:
        DataFrame mit Boolean-Spalten:
        - entry_long: True = gehe Long
        - entry_short: True = gehe Short
        - exit_long: True = schliesse Long
        - exit_short: True = schliesse Short
    """
    if use_filters:
        # Mit technischen Filtern (konservativ)
        long_signal = (
            (df["p_up"] > p_long_thr)
            & (df["atr_pct"].between(0.8, 6.0))
            & (df["Close"] > df["ema50"])
        )
        short_signal = (
            (df["p_up"] < p_short_thr)
            & (df["atr_pct"].between(0.8, 6.0))
            & (df["Close"] < df["ema50"])
        )
    else:
        # Ohne Filter (aggressiv) - nur ML-Signal
        long_signal = df["p_up"] > p_long_thr
        short_signal = df["p_up"] < p_short_thr

    return pd.DataFrame({
        "entry_long": long_signal,
        "entry_short": short_signal,
        "exit_long": short_signal,   # Exit Long = Entry Short
        "exit_short": long_signal      # Exit Short = Entry Long
    }, index=df.index)
