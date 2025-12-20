import unittest
import pandas as pd
import numpy as np
from src.backtest import SimpleBacktester


class TestSimpleBacktester(unittest.TestCase):
    """Unit-Tests für den SimpleBacktester."""

    def setUp(self):
        """Erstelle Test-Daten für jeden Test."""
        # Einfache Test-Daten: 10 Tage, konstanter Preis 100
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        self.df_flat = pd.DataFrame({
            "Open": [100.0] * 10,
            "High": [102.0] * 10,
            "Low": [98.0] * 10,
            "Close": [100.0] * 10,
            "Volume": [1000] * 10
        }, index=dates)

    def test_no_signals_flat_equity(self):
        """Wenn keine Signale → Equity sollte konstant bei 1.0 bleiben."""
        signals = pd.DataFrame({
            "entry_long": [False] * 10,
            "exit_long": [False] * 10
        }, index=self.df_flat.index)

        bt = SimpleBacktester(self.df_flat, fees_bps=0, slippage_bps=0)
        equity = bt.run(signals)

        # Alle Werte sollten 1.0 sein
        np.testing.assert_array_almost_equal(equity.values, np.ones(10))

    def test_perfect_trade_no_fees(self):
        """
        Entry bei 100, Exit bei 110 ohne Fees/Slippage.
        Erwarteter Return: 10%
        """
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df = pd.DataFrame({
            "Open": [100, 100, 110, 110, 110],
            "High": [102, 102, 112, 112, 112],
            "Low": [98, 98, 108, 108, 108],
            "Close": [100, 100, 110, 110, 110],
            "Volume": [1000] * 5
        }, index=dates)

        # Entry-Signal am Tag 0 → Kauf zu Open am Tag 1 (100)
        # Exit-Signal am Tag 2 → Verkauf zu Open am Tag 3 (110)
        signals = pd.DataFrame({
            "entry_long": [True, False, True, False, False],
            "exit_long": [False, False, False, False, False]
        }, index=dates)

        bt = SimpleBacktester(df, fees_bps=0, slippage_bps=0)
        equity = bt.run(signals)

        # Nach Exit am Tag 3: Equity sollte 1.1 sein (10% Gewinn)
        self.assertAlmostEqual(equity.iloc[-1], 1.1, places=5)

    def test_perfect_trade_with_fees(self):
        """
        Entry bei 100, Exit bei 110 mit 20 bps Fees (0.2%).
        Entry-Preis: 100 * (1 + 0.002) = 100.2
        Exit-Preis: 110 * (1 - 0.002) = 109.78
        Return: 109.78 / 100.2 - 1 ≈ 9.58%
        """
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df = pd.DataFrame({
            "Open": [100, 100, 110, 110, 110],
            "High": [102, 102, 112, 112, 112],
            "Low": [98, 98, 108, 108, 108],
            "Close": [100, 100, 110, 110, 110],
            "Volume": [1000] * 5
        }, index=dates)

        signals = pd.DataFrame({
            "entry_long": [True, False, True, False, False],
            "exit_long": [False, False, False, False, False]
        }, index=dates)

        bt = SimpleBacktester(df, fees_bps=20, slippage_bps=0)
        equity = bt.run(signals)

        # Equity sollte größer als 1.09 sein (ca. 9.58% Gewinn nach Fees)
        self.assertGreater(equity.iloc[-1], 1.09)
        self.assertLess(equity.iloc[-1], 1.11)

    def test_losing_trade(self):
        """
        Entry bei 100, Exit bei 90 (10% Verlust).
        Mit Fees sollte der Verlust größer sein.
        """
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df = pd.DataFrame({
            "Open": [100, 100, 90, 90, 90],
            "High": [102, 102, 92, 92, 92],
            "Low": [98, 98, 88, 88, 88],
            "Close": [100, 100, 90, 90, 90],
            "Volume": [1000] * 5
        }, index=dates)

        signals = pd.DataFrame({
            "entry_long": [True, False, True, False, False],
            "exit_long": [False, False, False, False, False]
        }, index=dates)

        bt = SimpleBacktester(df, fees_bps=20, slippage_bps=5)
        equity = bt.run(signals)

        # Equity sollte unter 1.0 sein (Verlust)
        self.assertLess(equity.iloc[-1], 1.0)

    def test_mark_to_market_during_position(self):
        """
        Teste Mark-to-Market während offener Position.
        Entry bei 100, Preis steigt auf 110, dann Exit.
        Während der Position sollte Equity mit Preis steigen.
        """
        dates = pd.date_range("2023-01-01", periods=6, freq="D")
        df = pd.DataFrame({
            "Open": [100, 100, 105, 110, 110, 110],
            "High": [102, 102, 107, 112, 112, 112],
            "Low": [98, 98, 103, 108, 108, 108],
            "Close": [100, 105, 110, 110, 110, 110],  # Preis steigt während Position
            "Volume": [1000] * 6
        }, index=dates)

        signals = pd.DataFrame({
            "entry_long": [True, False, False, True, False, False],  # Entry Tag 0, Exit Tag 3
            "exit_long": [False, False, False, False, False, False]
        }, index=dates)

        bt = SimpleBacktester(df, fees_bps=0, slippage_bps=0)
        equity = bt.run(signals)

        # Entry zu Open[1] = 100
        # Close[1] = 105 → Equity sollte ~1.05 sein
        # Close[2] = 110 → Equity sollte ~1.10 sein
        # Close[3] = 110 → Equity sollte ~1.10 sein (gleich bleibend)

        self.assertGreater(equity.iloc[2], equity.iloc[1])  # Steigt während Position
        self.assertAlmostEqual(equity.iloc[3], equity.iloc[2], places=5)  # Bleibt gleich bei konstantem Preis


    def test_multiple_trades(self):
        """
        Teste mehrere Trades hintereinander.
        """
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "Open": [100, 100, 110, 110, 110, 105, 105, 115, 115, 115],
            "High": [102] * 10,
            "Low": [98] * 10,
            "Close": [100, 110, 110, 110, 105, 105, 115, 115, 115, 115],
            "Volume": [1000] * 10
        }, index=dates)

        # Trade 1: Entry Tag 0, Exit Tag 2
        # Trade 2: Entry Tag 4, Exit Tag 6
        signals = pd.DataFrame({
            "entry_long": [True, False, False, False, True, False, False, False, False, False],
            "exit_long": [False, False, True, False, False, False, True, False, False, False]
        }, index=dates)

        bt = SimpleBacktester(df, fees_bps=0, slippage_bps=0)
        equity = bt.run(signals)

        # Nach 2 profitablen Trades sollte Equity > 1.0 sein
        self.assertGreater(equity.iloc[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
