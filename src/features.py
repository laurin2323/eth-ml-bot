import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    
    d["ema50"]  = EMAIndicator(d["Close"], 50).ema_indicator()
    d["ema200"] = EMAIndicator(d["Close"], 200).ema_indicator()
    d["rsi14"]  = RSIIndicator(d["Close"], 14).rsi()
    macd = MACD(d["Close"])
    d["macd_diff"] = macd.macd_diff()
    atr = AverageTrueRange(d["High"], d["Low"], d["Close"], 14)
    d["atr"] = atr.average_true_range()
    d["atr_pct"] = d["atr"] / d["Close"] * 100
    bb = BollingerBands(d["Close"], 20, 2)
    d["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / d["Close"]
    d["regime_bull"] = (d["Close"] > d["ema200"]).astype(int)
    d["ret1"] = np.log(d["Close"]).diff()
    return d.dropna()
