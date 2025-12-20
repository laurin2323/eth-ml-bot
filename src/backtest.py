import pandas as pd
import numpy as np

class SimpleBacktester:
    """
    Einfacher Backtester für Long-Only-Strategien mit Fees und Slippage.

    Logik:
    - Signal am Tag T-1 → Ausführung zu Open(T) (kein Look-Ahead-Bias)
    - Mark-to-Market Equity während offener Positionen
    - Fees und Slippage werden bei Entry und Exit berücksichtigt
    """
    def __init__(self, df, fees_bps=20, slippage_bps=5):
        self.df = df.copy()
        self.fees = fees_bps / 10000
        self.slip = slippage_bps / 10000

    def run(self, signals: pd.DataFrame):
        d = self.df.join(signals)
        position = 0
        entry_price = None
        equity_start = None
        equity = [1.0]

        for i in range(1, len(d)):
            row_prev = d.iloc[i-1]
            row = d.iloc[i]

            if position == 0 and row_prev["entry_long"]:
                position = 1
                entry_price = row["Open"] * (1 + self.slip + self.fees)
                equity_start = equity[-1]

            if position == 1:
                # Mark-to-Market: aktuelle Equity basierend auf Close-Preis
                current_equity = equity_start * (row["Close"] / entry_price)

                if row_prev["exit_long"]:
                    exit_price = row["Open"] * (1 - self.slip - self.fees)
                    equity.append(equity_start * (exit_price / entry_price))
                    position = 0
                    entry_price = None
                    equity_start = None
                else:
                    equity.append(current_equity)
            else:
                equity.append(equity[-1])

        return pd.Series(equity, index=d.index, name="equity")


class LongShortBacktester:
    """
    Backtester für Long/Short-Strategien.

    Logik:
    - Immer investiert: entweder Long (+1) oder Short (-1)
    - Short: Profitiert wenn Preis fällt
    - Signal am Tag T-1 → Ausführung zu Open(T)
    - Mark-to-Market während offener Positionen
    - Fees und Slippage bei jedem Trade (auch bei Wechsel Long<->Short)
    """
    def __init__(self, df, fees_bps=20, slippage_bps=5):
        self.df = df.copy()
        self.fees = fees_bps / 10000
        self.slip = slippage_bps / 10000

    def run(self, signals: pd.DataFrame):
        """
        Args:
            signals: DataFrame mit entry_long, entry_short, exit_long, exit_short

        Returns:
            Series mit Equity-Kurve
        """
        d = self.df.join(signals)
        position = 0  # 0=flat, 1=long, -1=short
        entry_price = None
        equity_start = None
        equity = [1.0]

        for i in range(1, len(d)):
            row_prev = d.iloc[i-1]
            row = d.iloc[i]

            # Entry Long (von Flat oder Short)
            if position != 1 and row_prev["entry_long"]:
                # Falls wir Short waren, erst schliessen
                if position == -1:
                    exit_price = row["Open"] * (1 + self.slip + self.fees)  # Short exit
                    pnl = (entry_price - exit_price) / entry_price  # Short PnL
                    equity_start = equity[-1] * (1 + pnl)

                # Dann Long eröffnen
                position = 1
                entry_price = row["Open"] * (1 + self.slip + self.fees)
                if equity_start is None:
                    equity_start = equity[-1]

            # Entry Short (von Flat oder Long)
            elif position != -1 and row_prev["entry_short"]:
                # Falls wir Long waren, erst schliessen
                if position == 1:
                    exit_price = row["Open"] * (1 - self.slip - self.fees)  # Long exit
                    pnl = (exit_price - entry_price) / entry_price
                    equity_start = equity[-1] * (1 + pnl)

                # Dann Short eröffnen
                position = -1
                entry_price = row["Open"] * (1 - self.slip - self.fees)  # Short entry
                if equity_start is None:
                    equity_start = equity[-1]

            # Mark-to-Market
            if position == 1:
                # Long: profitiert wenn Preis steigt
                current_equity = equity_start * (row["Close"] / entry_price)
                equity.append(current_equity)
            elif position == -1:
                # Short: profitiert wenn Preis fällt
                pnl = (entry_price - row["Close"]) / entry_price
                current_equity = equity_start * (1 + pnl)
                equity.append(current_equity)
            else:
                # Flat (sollte nie vorkommen bei Long/Short Policy)
                equity.append(equity[-1])

        return pd.Series(equity, index=d.index, name="equity")
