# ETH Trading Bot - ML-basierte Trading-Strategie

**Machine Learning Projekt zur algorithmischen Kryptowährung-Trading**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Experimental-yellow)

## 📋 Übersicht

Dieses Projekt implementiert einen **Machine Learning-basierten Trading Bot** für Ethereum (ETH-USD). Der Bot verwendet technische Indikatoren und Logistic Regression zur Vorhersage profitabler Trading-Opportunities.

**Wichtiger Hinweis:** Dieses Projekt dient zu **Lern- und Forschungszwecken**. Der Bot underperformt aktuell gegen eine simple Buy & Hold Strategie (siehe [Limitations](#limitations)).

## 🎯 Features

- **ML-Modelle:** Logistic Regression, Random Forest, Gradient Boosting
- **Technical Features:** EMAs, RSI, MACD, ATR, Bollinger Bands, Volume-Indikatoren
- **Backtesting:** Realistische Fees (20 bps) und Slippage (5 bps)
- **No Look-Ahead Bias:** Signal am Tag T → Execution zu Open(T+1)
- **Mark-to-Market:** Unrealized P&L tracking während offener Positionen
- **Live Predictions:** Tägliche Trading-Empfehlungen (BUY/SELL/HOLD)
- **Visualisierung:** Interaktive Plotly-Charts für alle Metriken

## 🚀 Quick Start

### Installation

```bash
# Repository klonen
git clone <your-repo-url>
cd ETH-TRADINGBOT-ML

# Virtual Environment erstellen
python -m venv .venv

# Aktivieren (Windows)
.venv\Scripts\activate

# Abhängigkeiten installieren
pip install -r requirements.txt
```

### Verwendung

#### 1. **Live Trading Empfehlung** (NEU!)
```bash
python -m src.predict_now
```
Gibt dir eine tagesaktuelle Empfehlung: BUY, SELL oder HOLD basierend auf dem ML-Modell.

**Beispiel-Output:**
```
======================================================================
TRADING SIGNAL
======================================================================
SIGNAL: HOLD
Empfehlung: Abwarten, keine Action
Begründung: ML-Prognose 39.7% nicht stark genug
             + ATR ausserhalb Range (6.08%)
             + Preis unter EMA50 ($2977.97 <= $3219.46)
```

#### 2. **Komplette Pipeline ausführen**
```bash
python -m src.run_pipeline
```
Trainiert das Modell und führt Backtest aus.

#### 3. **Experimente durchführen**
```bash
# Modell-Vergleich
python -m src.compare_models

# Feature-Vergleich
python -m src.compare_features

# Threshold-Optimierung
python -m src.optimize_thresholds

# Visualisierungen erstellen
python -m src.visualize_experiments
```

#### 4. **Buy & Hold Vergleich**
```bash
python -m src.compare_buy_hold
```
Vergleicht ML Bot gegen simple Buy & Hold Strategie.

#### 5. **Unit Tests**
```bash
pytest tests/ -v
```

## 📊 Performance

### Long-Only Strategy (Aktuelle Config)

**Test-Period:** 2023-01-01 bis 2025-12-20 (1085 Tage)

| Metrik | Wert |
|--------|------|
| **Return** | +42.30% |
| **Sharpe Ratio** | 0.673 |
| **CAGR** | 8.55% |
| **Max Drawdown** | -18.17% |
| **Trades** | 27 |

### vs. Buy & Hold

| Strategie | Return | Sharpe | CAGR | MaxDD |
|-----------|--------|--------|------|-------|
| **Buy & Hold** | **+147%** | 0.661 | **23.49%** | -63.79% |
| **ML Bot** | +42% | **0.673** | 8.55% | **-18.17%** |

**Ergebnis:** ML Bot underperformt um -105% vs Buy & Hold

⚠️ **Warum?** Siehe [Limitations](#limitations) Section

## 🧪 Experimente

### 1. Modell-Vergleich
- **LogReg:** Sharpe 0.673 ✅
- **Random Forest:** Sharpe 0.430
- **Gradient Boosting:** Sharpe -0.040 ❌

**Erkenntnis:** Einfache Modelle schlagen komplexe bei Finanzdaten

### 2. Feature-Vergleich
- **Basis (8 Features):** Sharpe 0.673 ✅
- **Mit Volumen (11 Features):** Sharpe 0.304

**Erkenntnis:** Mehr Features ≠ Besser

### 3. Threshold-Optimierung
- **Best:** p_entry_thr=0.55, p_exit_thr=0.1
- Validation-Set kann irreführend sein bei unterschiedlichen Marktphasen

Detaillierte Ergebnisse: [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)

## ⚠️ Limitations

### Hauptprobleme

1. **Market Exposure:** Nur 2.5% der Zeit investiert (27/1085 Tage)
2. **Long-Only:** Kann nicht von fallenden Märkten profitieren
3. **1-Day Labels:** Zu viel Noise für ML-Training
4. **Restriktive Rules:** ATR/EMA Filter blocken viele Trades
5. **Small Sample:** Nur 27 Trades → statistisch nicht signifikant

### Was funktioniert gut

✅ **Risikomanagement:** MaxDD -18% vs -64% bei Buy & Hold
✅ **Code-Qualität:** Saubere Pipeline, Tests, keine Biases
✅ **Methodologie:** Systematischer wissenschaftlicher Ansatz

### Verbesserungsvorschläge

**Implementiert (aber noch nicht optimal):**
- Long/Short Policy mit 5-Day Labels
- Relaxed Entry Rules

**Weitere Ideen:**
- Walk-Forward Optimization
- Position Sizing (Kelly Criterion)
- Multi-Timeframe Signals
- Alternative Strategies (Mean Reversion, Pairs Trading)

Ausführliche Analyse: [EXPERIMENT_RESULTS.md - Limitations Section](EXPERIMENT_RESULTS.md#limitations--kritische-analyse)

## 📁 Projekt-Struktur

```
ETH-TRADINGBOT-ML/
│
├── src/
│   ├── data.py              # Daten laden (yfinance)
│   ├── features.py          # Feature Engineering
│   ├── label.py             # Label-Generierung (1-day, 5-day)
│   ├── model.py             # ML Models (LogReg, RF, GBM)
│   ├── policy.py            # Entry/Exit Rules (Long-Only, Long/Short)
│   ├── backtest.py          # Backtester (Long-Only, Long/Short)
│   ├── eval.py              # Performance Metriken
│   ├── config.py            # Zentrale Konfiguration
│   ├── run_pipeline.py      # Haupt-Pipeline
│   ├── predict_now.py       # Live Trading Signals (NEU!)
│   ├── compare_buy_hold.py  # Buy & Hold Vergleich
│   ├── compare_improved.py  # Verbesserte Strategien
│   ├── compare_models.py    # Modell-Vergleich
│   ├── compare_features.py  # Feature-Vergleich
│   ├── optimize_thresholds.py # Grid Search
│   └── visualize_experiments.py # Plotting
│
├── tests/
│   └── test_backtest.py     # Unit Tests (6 Tests ✓)
│
├── plots/                   # HTML Visualisierungen
├── README.md
├── EXPERIMENT_RESULTS.md    # Detaillierte Experimente
├── requirements.txt
└── .gitignore
```

## 🔧 Konfiguration

Zentrale Config in [src/config.py](src/config.py):

```python
# Trading Policy Thresholds
P_ENTRY_THR = 0.55  # Entry wenn p_up > 0.55
P_EXIT_THR = 0.1    # Exit wenn p_up < 0.1

# Features (8 Basis-Features ohne Volumen)
FEATURES = [
    "ema50", "ema200", "rsi14", "macd_diff",
    "atr_pct", "bb_width", "regime_bull", "ret1"
]
```

## 📚 Technologien

- **Python 3.8+**
- **Pandas:** Datenverarbeitung
- **Scikit-Learn:** ML Models
- **TA-Lib:** Technische Indikatoren
- **yfinance:** Krypto-Daten
- **Plotly:** Interaktive Charts
- **Pytest:** Unit Testing

## 📈 Live Trading

### Tägliche Nutzung

1. Jeden Morgen ausführen:
   ```bash
   python -m src.predict_now
   ```

2. Empfehlung beachten:
   - **BUY:** Long Position eröffnen
   - **SELL:** Bestehende Position schliessen
   - **HOLD:** Keine Action

3. **Wichtig:**
   - Dies ist KEINE Anlageberatung
   - Immer eigenes Risikomanagement beachten
   - Bot hat aktuell negative Alpha vs Buy & Hold

### Modell-Retraining

Für produktiven Einsatz sollte das Modell regelmässig (z.B. wöchentlich) neu trainiert werden:

```python
# In predict_now.py
# Aktuell: Training bis 2023-01-01 (VERALTET!)
# Besser: Rolling Window Training
```

## 🧪 Testing

```bash
# Alle Tests
pytest tests/ -v

# Mit Coverage
pytest tests/ --cov=src --cov-report=html
```

**Test-Coverage:** 6/6 Tests passing ✓

## 🤝 Contributing

Dies ist ein Lernprojekt. Verbesserungsvorschläge sind willkommen:

1. Fork the repo
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push (`git push origin feature/improvement`)
5. Open Pull Request

## 📝 License

MIT License - Verwendung auf eigenes Risiko!

## ⚠️ Disclaimer

**WICHTIG:**
- Dieses Projekt dient zu **Lern- und Forschungszwecken**
- Keine Anlageberatung
- Trading mit Kryptowährungen ist hochriskant
- Vergangene Performance garantiert keine zukünftigen Ergebnisse
- Der Bot underperformt aktuell gegen Buy & Hold
- Nur mit Geld handeln, das du bereit bist zu verlieren

## 📧 Kontakt

**Autor:** ML Trading Bot Projekt
**Erstellt:** 2025-12-20
**Framework:** FHNW Machine Learning Kurs

---

**Happy Trading!** 🚀📈

(Aber bitte erstmal nur auf Papier 📝)
