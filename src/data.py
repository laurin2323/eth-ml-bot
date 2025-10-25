import yfinance as yf
import pandas as pd

def download_eth_1d(start="2019-01-01", end=None, ticker="ETH-USD"):
    df = yf.download(
        tickers=ticker,
        interval="1d",
        start=start,
        end=end,
        auto_adjust=False,  # explizit, um Überraschungen zu vermeiden
        threads=True
    )
    # MultiIndex -> flach
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Einheitliche Spaltennamen
    df = df.rename(columns=str.title)  # Open High Low Close Volume Adj Close (falls vorhanden)
    # Nur die Spalten, die wir brauchen
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].dropna()
    df.index.name = "Date"
    return df

