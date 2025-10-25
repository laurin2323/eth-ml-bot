from src.data import download_eth_1d
from src.features import add_features
from src.label import make_label
from src.model import train_logreg, infer_proba
from src.policy import ml_policy
from src.backtest import SimpleBacktester
from src.eval import returns_from_equity, sharpe, max_drawdown

def main():
    # 1) Daten
    df = download_eth_1d(start="2019-01-01")

    # 2) Features
    feat = add_features(df)

    # 3) Label
    lab = make_label(feat, fee_buffer=0.0025)

    # 4) Zeitbasierter Split
    split_date = "2023-01-01"
    train = lab.loc[:split_date]
    test  = lab.loc[split_date:]

    # 5) Modell
    model = train_logreg(train)

    # 6) Inferenz + Policy
    test_pred = infer_proba(model, test)
    signals = ml_policy(test_pred, p_thr=0.55)

    # 7) Backtest
    bt = SimpleBacktester(test_pred)
    equity = bt.run(signals)

    # 8) Kennzahlen
    ret = returns_from_equity(equity)
    print("Sharpe:", round(sharpe(ret, periods=252), 2))
    print("MaxDD:", f"{round(max_drawdown(equity)*100, 1)} %")

if __name__ == "__main__":
    main()
