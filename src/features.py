import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator, VolumeWeightedAveragePrice

def add_features(df: pd.DataFrame, include_volume=True) -> pd.DataFrame:
    """
    Berechnet technische Indikatoren als Features.

    Args:
        df: DataFrame mit OHLCV-Daten
        include_volume: Wenn True, werden Volumen-Indikatoren hinzugefügt (default: True)

    Returns:
        DataFrame mit zusätzlichen Feature-Spalten
    """
    d = df.copy()

    # Trend-Indikatoren
    d["ema50"]  = EMAIndicator(d["Close"], 50).ema_indicator()
    d["ema200"] = EMAIndicator(d["Close"], 200).ema_indicator()

    # Momentum-Indikatoren
    d["rsi14"]  = RSIIndicator(d["Close"], 14).rsi()
    macd = MACD(d["Close"])
    d["macd_diff"] = macd.macd_diff()

    # Volatilitäts-Indikatoren
    atr = AverageTrueRange(d["High"], d["Low"], d["Close"], 14)
    d["atr"] = atr.average_true_range()
    d["atr_pct"] = d["atr"] / d["Close"] * 100
    bb = BollingerBands(d["Close"], 20, 2)
    d["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / d["Close"]

    # Regime
    d["regime_bull"] = (d["Close"] > d["ema200"]).astype(int)

    # Returns
    d["ret1"] = np.log(d["Close"]).diff()

    # Volumen-Indikatoren (optional)
    if include_volume and "Volume" in d.columns:
        # On-Balance Volume (OBV)
        obv = OnBalanceVolumeIndicator(d["Close"], d["Volume"])
        d["obv"] = obv.on_balance_volume()
        d["obv_ema"] = d["obv"].ewm(span=20).mean()  # Geglättet

        # Money Flow Index (MFI) - wie RSI, aber mit Volumen
        mfi = MFIIndicator(d["High"], d["Low"], d["Close"], d["Volume"], 14)
        d["mfi"] = mfi.money_flow_index()

        # Volume Ratio (aktuelles Volumen / durchschnittliches Volumen)
        d["vol_sma20"] = d["Volume"].rolling(20).mean()
        d["vol_ratio"] = d["Volume"] / d["vol_sma20"]

    return d.dropna()
