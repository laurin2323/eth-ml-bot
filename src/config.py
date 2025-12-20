# Threshold-Konfiguration
P_ENTRY_THR = 0.55
P_EXIT_THR  = 0.1

# Sweep-Grid
ENTRY_THR_GRID = [i / 100 for i in range(30, 71, 5)]  # 0.30–0.70

# Feature-Liste für ML-Modell (Basis ohne Volumen)
FEATURES_BASE = [
    "ema50",
    "ema200",
    "rsi14",
    "macd_diff",
    "atr_pct",
    "bb_width",
    "regime_bull",
    "ret1"
]

# Feature-Liste MIT Volumen-Indikatoren
FEATURES_WITH_VOLUME = FEATURES_BASE + [
    "obv_ema",    # On-Balance Volume (geglättet)
    "mfi",        # Money Flow Index
    "vol_ratio"   # Volume Ratio
]

# Default: Nutze Basis-Features (wie vorher)
FEATURES = FEATURES_BASE
