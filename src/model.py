import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

FEATURES = ["ema50","ema200","rsi14","macd_diff","atr_pct","bb_width","regime_bull","ret1"]

def train_logreg(train_df):
    X = train_df[FEATURES].values
    y = train_df["y"].values
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=300))])
    pipe.fit(X, y)
    return pipe

def infer_proba(model, df):
    proba = model.predict_proba(df[FEATURES].values)[:,1]
    out = df.copy()
    out["p_up"] = proba
    return out
