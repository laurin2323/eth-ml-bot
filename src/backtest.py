import pandas as pd

class SimpleBacktester:
    def __init__(self, df, fees_bps=20, slippage_bps=5):
        self.df = df.copy()
        self.fees = fees_bps / 10000
        self.slip = slippage_bps / 10000

    def run(self, signals: pd.DataFrame):
        d = self.df.join(signals)
        position = 0
        entry_price = None
        equity = [1.0]

        for i in range(1, len(d)):
            row_prev = d.iloc[i-1]
            row = d.iloc[i]

            if position == 0 and row_prev["entry_long"]:
                position = 1
                entry_price = row["Open"] * (1 + self.slip + self.fees)

            if position == 1:
                if row_prev["exit_long"]:
                    exit_price = row["Open"] * (1 - self.slip - self.fees)
                    equity.append(equity[-1] * (exit_price / entry_price))
                    position = 0
                    entry_price = None
                else:
                    equity.append(equity[-1])
            else:
                equity.append(equity[-1])

        return pd.Series(equity, index=d.index, name="equity")
