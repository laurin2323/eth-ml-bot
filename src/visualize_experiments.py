"""
Erstellt umfassende Visualisierungen aller Experimente.
"""

from src.data import download_eth_1d
from src.features import add_features
from src.label import make_label
from src.model import train_logreg, train_random_forest, train_gradient_boosting, infer_proba
from src.policy import ml_policy
from src.backtest import SimpleBacktester
from src.eval import returns_from_equity, sharpe, max_drawdown, cagr
from src.config import P_ENTRY_THR, P_EXIT_THR, FEATURES_BASE, FEATURES_WITH_VOLUME
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def create_plots_dir():
    """Erstellt plots-Verzeichnis falls nicht vorhanden."""
    Path("plots").mkdir(parents=True, exist_ok=True)


def test_model_with_equity(model_name, model, test, p_entry_thr, p_exit_thr):
    """Testet Modell und gibt Equity-Kurve zurück."""
    test_pred = infer_proba(model, test)
    signals_df = ml_policy(test_pred, p_entry_thr=p_entry_thr, p_exit_thr=p_exit_thr)
    signals = signals_df[["entry_long", "exit_long"]].astype(int)

    bt = SimpleBacktester(test_pred)
    equity = bt.run(signals)
    ret = returns_from_equity(equity)

    return {
        "name": model_name,
        "equity": equity,
        "returns": ret,
        "sharpe": sharpe(ret, periods=252),
        "cagr": cagr(equity) * 100,
        "maxdd": max_drawdown(equity) * 100
    }


def plot_model_comparison(results):
    """Plot: Vergleich verschiedener Modelle."""
    # Subplot: Equity-Kurven
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Equity-Kurven", "Sharpe Ratio", "CAGR %", "Max Drawdown %"),
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )

    # Equity-Kurven
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, res in enumerate(results):
        fig.add_trace(
            go.Scatter(x=res["equity"].index, y=res["equity"],
                      name=res["name"], line=dict(color=colors[i])),
            row=1, col=1
        )

    # Metriken
    names = [r["name"] for r in results]
    sharpes = [r["sharpe"] for r in results]
    cagrs = [r["cagr"] for r in results]
    maxdds = [r["maxdd"] for r in results]

    fig.add_trace(
        go.Bar(x=names, y=sharpes, name="Sharpe", marker_color='lightblue'),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=names, y=cagrs, name="CAGR", marker_color='lightgreen'),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=names, y=maxdds, name="MaxDD", marker_color='salmon'),
        row=2, col=2
    )

    fig.update_xaxes(title_text="Datum", row=1, col=1)
    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_yaxes(title_text="Sharpe", row=1, col=2)
    fig.update_yaxes(title_text="CAGR %", row=2, col=1)
    fig.update_yaxes(title_text="MaxDD %", row=2, col=2)

    fig.update_layout(
        height=800,
        title_text="Modell-Vergleich: Logistic Regression vs Random Forest vs Gradient Boosting",
        showlegend=True
    )

    create_plots_dir()
    fig.write_html("plots/experiment_1_model_comparison.html")
    print("  -> Gespeichert: plots/experiment_1_model_comparison.html")


def plot_feature_comparison(results):
    """Plot: Vergleich mit/ohne Volumen-Features."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Equity-Kurven", "Metriken-Vergleich"),
        specs=[[{"type": "scatter"}, {"type": "bar"}]]
    )

    # Equity-Kurven
    colors = ['#1f77b4', '#ff7f0e']
    for i, res in enumerate(results):
        fig.add_trace(
            go.Scatter(x=res["equity"].index, y=res["equity"],
                      name=res["name"], line=dict(color=colors[i])),
            row=1, col=1
        )

    # Metriken
    names = [r["name"] for r in results]
    metrics_data = {
        "Sharpe": [r["sharpe"] for r in results],
        "CAGR %": [r["cagr"] for r in results],
        "MaxDD %": [abs(r["maxdd"]) for r in results]
    }

    x_pos = [0, 1]
    colors_bar = ['lightblue', 'lightgreen', 'salmon']
    for i, (metric, values) in enumerate(metrics_data.items()):
        fig.add_trace(
            go.Bar(x=[n + f" ({metric})" for n in names], y=values,
                  name=metric, marker_color=colors_bar[i]),
            row=1, col=2
        )

    fig.update_xaxes(title_text="Datum", row=1, col=1)
    fig.update_yaxes(title_text="Equity", row=1, col=1)

    fig.update_layout(
        height=500,
        title_text="Feature-Vergleich: Basis vs Mit Volumen",
        showlegend=True
    )

    create_plots_dir()
    fig.write_html("plots/experiment_3_feature_comparison.html")
    print("  -> Gespeichert: plots/experiment_3_feature_comparison.html")


def plot_returns_distribution(equity, model_name):
    """Plot: Histogramm der täglichen Returns."""
    returns = returns_from_equity(equity)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name="Returns",
        marker_color='steelblue'
    ))

    fig.update_layout(
        title=f"Verteilung der täglichen Returns - {model_name}",
        xaxis_title="Täglicher Return",
        yaxis_title="Häufigkeit",
        showlegend=False
    )

    create_plots_dir()
    fig.write_html(f"plots/returns_distribution_{model_name.lower().replace(' ', '_')}.html")
    print(f"  -> Gespeichert: plots/returns_distribution_{model_name.lower().replace(' ', '_')}.html")


def plot_drawdown_underwater(equity, model_name):
    """Plot: Underwater/Drawdown Chart."""
    cummax = equity.cummax()
    drawdown = (equity / cummax - 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown,
        fill='tozeroy',
        name='Drawdown',
        line=dict(color='red')
    ))

    fig.update_layout(
        title=f"Drawdown-Analyse - {model_name}",
        xaxis_title="Datum",
        yaxis_title="Drawdown (%)",
        showlegend=False
    )

    create_plots_dir()
    fig.write_html(f"plots/drawdown_{model_name.lower().replace(' ', '_')}.html")
    print(f"  -> Gespeichert: plots/drawdown_{model_name.lower().replace(' ', '_')}.html")


def main():
    print("=" * 70)
    print("VISUALISIERUNG ALLER EXPERIMENTE")
    print("=" * 70)

    # Daten laden
    print("\n[1/4] Lade Daten und bereite vor...")
    df = download_eth_1d(start="2019-01-01")
    feat = add_features(df, include_volume=False)
    lab = make_label(feat, fee_buffer=0.0025)

    feat_vol = add_features(df, include_volume=True)
    lab_vol = make_label(feat_vol, fee_buffer=0.0025)

    # Split
    split_date = "2023-01-01"
    train = lab.loc[:split_date]
    test = lab.loc[split_date:]
    train_vol = lab_vol.loc[:split_date]
    test_vol = lab_vol.loc[split_date:]

    # Experiment 1 & 2: Modell-Vergleich
    print("\n[2/4] Erstelle Modell-Vergleich Plots...")
    model_results = []

    print("  -> Trainiere Logistic Regression...")
    model_lr = train_logreg(train)
    model_results.append(test_model_with_equity("Logistic Regression", model_lr, test, P_ENTRY_THR, P_EXIT_THR))

    print("  -> Trainiere Random Forest...")
    model_rf = train_random_forest(train, n_estimators=100, max_depth=10)
    model_results.append(test_model_with_equity("Random Forest", model_rf, test, P_ENTRY_THR, P_EXIT_THR))

    print("  -> Trainiere Gradient Boosting...")
    model_gb = train_gradient_boosting(train, n_estimators=100, max_depth=5)
    model_results.append(test_model_with_equity("Gradient Boosting", model_gb, test, P_ENTRY_THR, P_EXIT_THR))

    plot_model_comparison(model_results)

    # Experiment 3: Feature-Vergleich
    print("\n[3/4] Erstelle Feature-Vergleich Plots...")
    feature_results = []

    # Basis-Features
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    print("  -> Teste Basis-Features...")
    model_base = train_logreg(train)
    feature_results.append(test_model_with_equity("Basis (8 Features)", model_base, test, P_ENTRY_THR, P_EXIT_THR))

    print("  -> Teste mit Volumen-Features...")
    X_train = train_vol[FEATURES_WITH_VOLUME].values
    y_train = train_vol["y"].values
    model_vol = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=300))
    ])
    model_vol.fit(X_train, y_train)

    X_test = test_vol[FEATURES_WITH_VOLUME].values
    proba = model_vol.predict_proba(X_test)[:, 1]
    test_vol_pred = test_vol.copy()
    test_vol_pred["p_up"] = proba

    signals_vol = ml_policy(test_vol_pred, p_entry_thr=P_ENTRY_THR, p_exit_thr=P_EXIT_THR)
    bt_vol = SimpleBacktester(test_vol_pred)
    equity_vol = bt_vol.run(signals_vol[["entry_long", "exit_long"]].astype(int))
    ret_vol = returns_from_equity(equity_vol)

    feature_results.append({
        "name": "Mit Volumen (11 Features)",
        "equity": equity_vol,
        "returns": ret_vol,
        "sharpe": sharpe(ret_vol, periods=252),
        "cagr": cagr(equity_vol) * 100,
        "maxdd": max_drawdown(equity_vol) * 100
    })

    plot_feature_comparison(feature_results)

    # Zusätzliche Analysen für bestes Modell
    print("\n[4/4] Erstelle Detail-Analysen für bestes Modell...")
    best_model = model_results[0]  # Logistic Regression
    plot_returns_distribution(best_model["equity"], best_model["name"])
    plot_drawdown_underwater(best_model["equity"], best_model["name"])

    print("\n" + "=" * 70)
    print("FERTIG! Alle Plots gespeichert in: plots/")
    print("=" * 70)
    print("\nErstelte Plots:")
    print("  1. experiment_1_model_comparison.html")
    print("  2. experiment_3_feature_comparison.html")
    print("  3. returns_distribution_logistic_regression.html")
    print("  4. drawdown_logistic_regression.html")
    print("\nÖffne die HTML-Dateien im Browser für interaktive Visualisierungen!")


if __name__ == "__main__":
    main()
