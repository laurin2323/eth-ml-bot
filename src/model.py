import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from src.config import FEATURES

def train_logreg(train_df):
    """
    Trainiert ein Logistic Regression Modell mit StandardScaler.

    Args:
        train_df: DataFrame mit Features (aus FEATURES) und Label 'y'

    Returns:
        Sklearn Pipeline mit Scaler und Classifier
    """
    X = train_df[FEATURES].values
    y = train_df["y"].values
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=300))])
    pipe.fit(X, y)
    return pipe

def train_random_forest(train_df, n_estimators=100, max_depth=10):
    """
    Trainiert ein Random Forest Modell mit StandardScaler.

    Args:
        train_df: DataFrame mit Features (aus FEATURES) und Label 'y'
        n_estimators: Anzahl der Bäume (default: 100)
        max_depth: Maximale Tiefe der Bäume (default: 10)

    Returns:
        Sklearn Pipeline mit Scaler und Random Forest Classifier
    """
    X = train_df[FEATURES].values
    y = train_df["y"].values
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1  # Nutze alle CPU-Cores
        ))
    ])
    pipe.fit(X, y)
    return pipe

def train_gradient_boosting(train_df, n_estimators=100, max_depth=5, learning_rate=0.1):
    """
    Trainiert ein Gradient Boosting Modell (ähnlich zu XGBoost).

    Args:
        train_df: DataFrame mit Features (aus FEATURES) und Label 'y'
        n_estimators: Anzahl der Boosting-Stufen (default: 100)
        max_depth: Maximale Tiefe der Bäume (default: 5)
        learning_rate: Lernrate (default: 0.1)

    Returns:
        Sklearn Pipeline mit Scaler und Gradient Boosting Classifier
    """
    X = train_df[FEATURES].values
    y = train_df["y"].values
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        ))
    ])
    pipe.fit(X, y)
    return pipe

def infer_proba(model, df):
    """
    Berechnet Wahrscheinlichkeiten für die positive Klasse (y=1).

    Args:
        model: Trainiertes Sklearn-Modell
        df: DataFrame mit Features (aus FEATURES)

    Returns:
        DataFrame mit zusätzlicher Spalte 'p_up' (Wahrscheinlichkeit für Aufwärtsbewegung)
    """
    proba = model.predict_proba(df[FEATURES].values)[:,1]
    out = df.copy()
    out["p_up"] = proba
    return out
