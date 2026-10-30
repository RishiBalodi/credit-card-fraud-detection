"""
Exploratory Data Analysis and Visualizations Module
==================================================
Generates high-resolution publication-quality Matplotlib/Seaborn plots
for reports and rich interactive Plotly figures for the Streamlit dashboard.
"""

import os
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


# Styling Constants for Fintech Dark Theme
COLOR_LEGIT = "#10b981"    # Emerald Green
COLOR_FRAUD = "#ef4444"    # Crimson Red
COLOR_WARNING = "#f59e0b"  # Amber
COLOR_PRIMARY = "#38bdf8"  # Cyan / Electric Blue
COLOR_ACCENT = "#818cf8"   # Indigo Accent
DARK_BG = "#0f172a"        # Deep Slate


# -------------------------------------------------------------------------
# 1. MATPLOTLIB / SEABORN STATIC SAVED PLOTS (Saved into reports/figures/)
# -------------------------------------------------------------------------

def plot_total_legit_vs_fraud(df: pd.DataFrame, output_path: str):
    """Visualizes total legitimate vs fraudulent transactions count & percentage."""
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df["is_fraud"].value_counts().rename({0: "Legitimate", 1: "Fraudulent"})
    percentages = (counts / len(df)) * 100

    bars = ax.bar(counts.index, counts.values, color=[COLOR_LEGIT, COLOR_FRAUD], width=0.45, edgecolor="none")
    ax.set_title("Total Legitimate vs Fraudulent Transactions", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Transaction Count", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, pct, count in zip(bars, percentages.values, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(counts.values) * 0.015),
            f"{count:,}\n({pct:.2f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_fraud_by_category(df: pd.DataFrame, output_path: str):
    """Visualizes fraud transaction distribution and fraud rate by merchant category."""
    if "category" not in df.columns:
        return

    cat_stats = df.groupby("category")["is_fraud"].agg(
        total="count",
        fraud_count="sum",
        fraud_rate=lambda x: (x.sum() / len(x)) * 100.0
    ).reset_index().sort_values(by="fraud_count", ascending=False)

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax2 = ax1.twinx()

    x = np.arange(len(cat_stats))
    width = 0.4

    b1 = ax1.bar(x - width/2, cat_stats["fraud_count"], width, label="Fraud Count", color=COLOR_FRAUD, alpha=0.85)
    b2 = ax2.bar(x + width/2, cat_stats["fraud_rate"], width, label="Fraud Rate (%)", color="#0284c7", alpha=0.85)

    ax1.set_xlabel("Merchant Category", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Number of Fraud Transactions", fontsize=11, color=COLOR_FRAUD)
    ax2.set_ylabel("Fraud Rate (%)", fontsize=11, color="#0284c7")
    ax1.set_title("Fraud Distribution and Fraud Rate by Merchant Category", fontsize=13, fontweight="bold", pad=12)

    ax1.set_xticks(x)
    ax1.set_xticklabels(cat_stats["category"], rotation=45, ha="right", fontsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_fraud_by_amount(df: pd.DataFrame, output_path: str):
    """Visualizes spending distribution comparison between legitimate and fraud transactions."""
    fig, ax = plt.subplots(figsize=(8, 5))
    legit_amt = df[df["is_fraud"] == 0]["amt"]
    fraud_amt = df[df["is_fraud"] == 1]["amt"]

    sns.boxplot(
        x="is_fraud",
        y="amt",
        data=df,
        ax=ax,
        palette=[COLOR_LEGIT, COLOR_FRAUD],
        showfliers=False
    )
    ax.set_xticklabels(["Legitimate (0)", "Fraudulent (1)"], fontsize=11)
    ax.set_xlabel("Transaction Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("Transaction Amount ($)", fontsize=11, fontweight="bold")
    ax.set_title("Transaction Amount Distribution: Legitimate vs Fraudulent", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    med_legit = legit_amt.median()
    med_fraud = fraud_amt.median()
    ax.text(0, med_legit, f" Median: ${med_legit:.2f}", va="bottom", ha="left", fontweight="bold", color="darkgreen")
    ax.text(1, med_fraud, f" Median: ${med_fraud:.2f}", va="bottom", ha="left", fontweight="bold", color="darkred")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_fraud_by_gender(df: pd.DataFrame, output_path: str):
    """Visualizes fraud distribution by cardholder gender."""
    if "gender" not in df.columns:
        return

    gender_stats = df.groupby("gender")["is_fraud"].agg(
        total="count",
        fraud_count="sum",
        fraud_rate=lambda x: (x.sum() / len(x)) * 100.0
    ).reset_index()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(gender_stats["gender"], gender_stats["fraud_count"], color=["#0284c7", COLOR_ACCENT], width=0.4)
    ax.set_title("Fraudulent Transactions by Cardholder Gender", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Gender", fontsize=10, fontweight="bold")
    ax.set_ylabel("Fraud Count", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, rate, count in zip(bars, gender_stats["fraud_rate"], gender_stats["fraud_count"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{count} ({rate:.2f}%)",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_fraud_by_state(df: pd.DataFrame, output_path: str):
    """Visualizes fraud occurrences across top states."""
    if "state" not in df.columns:
        return

    state_fraud = df[df["is_fraud"] == 1]["state"].value_counts().head(10)
    if state_fraud.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(state_fraud.index[::-1], state_fraud.values[::-1], color=COLOR_WARNING, height=0.55)
    ax.set_title("Top 10 States by Fraudulent Transaction Volume", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Fraud Count", fontsize=10, fontweight="bold")
    ax.set_ylabel("State", fontsize=10, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    for bar in bars:
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, f"{int(bar.get_width())}",
                va="center", ha="left", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_amount_distribution(df: pd.DataFrame, output_path: str):
    """Visualizes full transaction amount distribution histogram & KDE."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    capped_amt = df["amt"].clip(upper=df["amt"].quantile(0.99))
    sns.histplot(capped_amt, bins=40, kde=True, ax=ax, color="#0284c7")
    ax.set_title("Transaction Amount Distribution (Up to 99th Percentile)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Transaction Amount ($)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_fraud_by_hour(df: pd.DataFrame, output_path: str):
    """Visualizes fraudulent transactions by hour of day."""
    if "trans_hour" not in df.columns and "trans_date_trans_time" in df.columns:
        df["trans_hour"] = pd.to_datetime(df["trans_date_trans_time"]).dt.hour

    if "trans_hour" not in df.columns:
        return

    hour_fraud = df.groupby("trans_hour")["is_fraud"].sum()
    all_hours = pd.Series(0, index=np.arange(24))
    hour_fraud = all_hours.add(hour_fraud, fill_value=0)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(hour_fraud.index, hour_fraud.values, color=COLOR_FRAUD, width=0.6, alpha=0.85)
    ax.set_title("Fraudulent Transactions by Hour of the Day (0-23h)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Hour of Day (24-Hour Format)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Fraud Count", fontsize=10)
    ax.set_xticks(range(0, 24))
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax.axvspan(-0.5, 5.5, color="gray", alpha=0.15, label="Nighttime Window (22h-05h)")
    ax.axvspan(21.5, 23.5, color="gray", alpha=0.15)
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, output_path: str):
    """Generates correlation heatmap across numerical spending & behavioral features."""
    num_cols = [c for c in ["amt", "is_fraud", "trans_hour", "is_night", "distance_km", "age", "city_pop"] if c in df.columns]
    if len(num_cols) < 2:
        return

    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=ax,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5
    )
    ax.set_title("Correlation Heatmap of Numerical Features", fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_roc_curve(roc_data: Dict[str, Any], output_path: str):
    """Generates a professional publication-grade ROC curve plot."""
    fpr = roc_data["fpr"]
    tpr = roc_data["tpr"]
    roc_auc = roc_data.get("roc_auc", 0.0)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(fpr, tpr, color="#0284c7", lw=2.5, label=f"Decision Tree (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.5000)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_confusion_matrix_figure(metrics: Dict[str, Any], output_path: str):
    """Generates confusion matrix heatmap with TP, TN, FP, FN counts."""
    tn = metrics.get("true_negatives", 0)
    fp = metrics.get("false_positives", 0)
    fn = metrics.get("false_negatives", 0)
    tp = metrics.get("true_positives", 0)

    cm = np.array([[tn, fp], [fn, tp]])
    labels = [
        [f"TN\n{tn:,}", f"FP\n{fp:,}"],
        [f"FN\n{fn:,}", f"TP\n{tp:,}"]
    ]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=["Predicted Legit", "Predicted Fraud"],
        yticklabels=["Actual Legit", "Actual Fraud"],
        annot_kws={"fontsize": 12, "fontweight": "bold"}
    )
    ax.set_title("Confusion Matrix (Test Partition)", fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_feature_importance_figure(fi_df: pd.DataFrame, output_path: str):
    """Generates horizontal bar chart of Decision Tree feature importances."""
    top_fi = fi_df.head(10).sort_values(by="importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(top_fi["feature"], top_fi["importance"], color="#0284c7", height=0.55)
    ax.set_title("Decision Tree Feature Importance Rankings", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Importance Score", fontsize=10, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    for bar, val in zip(bars, top_fi["importance"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                va="center", ha="left", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def generate_all_saved_plots(
    df: pd.DataFrame,
    metrics: Dict[str, Any],
    roc_data: Dict[str, Any],
    fi_df: pd.DataFrame,
    output_dir: str
):
    """Generates and writes all analytical figures to disk in reports/figures/."""
    os.makedirs(output_dir, exist_ok=True)
    plot_total_legit_vs_fraud(df, os.path.join(output_dir, "fraud_vs_legit.png"))
    plot_fraud_by_category(df, os.path.join(output_dir, "fraud_distribution_by_category.png"))
    plot_fraud_by_amount(df, os.path.join(output_dir, "fraud_distribution_by_amount.png"))
    plot_fraud_by_gender(df, os.path.join(output_dir, "fraud_distribution_by_gender.png"))
    plot_fraud_by_state(df, os.path.join(output_dir, "fraud_distribution_by_state.png"))
    plot_amount_distribution(df, os.path.join(output_dir, "amount_distribution.png"))
    plot_fraud_by_hour(df, os.path.join(output_dir, "fraud_distribution_by_hour.png"))
    plot_correlation_heatmap(df, os.path.join(output_dir, "correlation_heatmap.png"))
    plot_roc_curve(roc_data, os.path.join(output_dir, "auc_roc_curve.png"))
    plot_confusion_matrix_figure(metrics, os.path.join(output_dir, "confusion_matrix.png"))
    plot_feature_importance_figure(fi_df, os.path.join(output_dir, "feature_importance.png"))


# -------------------------------------------------------------------------
# 2. PLOTLY RICH INTERACTIVE CHARTS (For Responsive Streamlit Dashboard)
# -------------------------------------------------------------------------

def create_interactive_class_distribution(df: pd.DataFrame) -> go.Figure:
    """Plotly interactive donut chart with center annotations."""
    counts = df["is_fraud"].value_counts()
    legit_cnt = int(counts.get(0, 0))
    fraud_cnt = int(counts.get(1, 0))
    total_cnt = legit_cnt + fraud_cnt
    fraud_pct = (fraud_cnt / total_cnt) * 100.0 if total_cnt > 0 else 0.0

    fig = go.Figure(data=[go.Pie(
        labels=["Legitimate", "Fraudulent"],
        values=[legit_cnt, fraud_cnt],
        hole=0.68,
        marker=dict(colors=[COLOR_LEGIT, COLOR_FRAUD], line=dict(color="#0f172a", width=2)),
        textinfo="percent+value",
        textfont=dict(size=13, color="#ffffff"),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>"
    )])

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=15, r=15, t=15, b=15),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, font=dict(color="#94a3b8")),
        annotations=[dict(
            text=f"<b>{fraud_pct:.2f}%</b><br><span style='font-size:11px;color:#94a3b8'>Fraud Rate</span>",
            x=0.5, y=0.5, font=dict(size=18, color="#ef4444"), showarrow=False
        )]
    )
    return fig


def create_interactive_category_chart(df: pd.DataFrame) -> go.Figure:
    """Plotly horizontal bar chart of fraud counts across merchant categories."""
    if "category" not in df.columns:
        return go.Figure()

    cat_stats = df.groupby("category")["is_fraud"].agg(["count", "sum"]).reset_index()
    cat_stats.columns = ["category", "total", "fraud_count"]
    cat_stats["fraud_rate"] = (cat_stats["fraud_count"] / cat_stats["total"]) * 100.0
    cat_stats = cat_stats.sort_values(by="fraud_count", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=cat_stats["category"],
        x=cat_stats["fraud_count"],
        orientation="h",
        marker=dict(
            color=cat_stats["fraud_count"],
            colorscale=[[0, "#38bdf8"], [0.5, "#f59e0b"], [1.0, "#ef4444"]],
            line=dict(color="rgba(255,255,255,0.1)", width=1)
        ),
        hovertemplate="<b>%{y}</b><br>Fraud Incidents: %{x}<br>Total Transactions: %{customdata[0]:,}<br>Fraud Rate: %{customdata[1]:.2f}%<extra></extra>",
        customdata=np.stack((cat_stats["total"], cat_stats["fraud_rate"]), axis=-1)
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Fraudulent Transaction Count",
        margin=dict(l=10, r=15, t=10, b=15),
        font=dict(color="#94a3b8"),
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b")
    )
    return fig


def create_interactive_hourly_fraud_chart(df: pd.DataFrame) -> go.Figure:
    """Plotly interactive hourly risk distribution highlighting nighttime danger zone."""
    if "trans_hour" not in df.columns and "trans_date_trans_time" in df.columns:
        df["trans_hour"] = pd.to_datetime(df["trans_date_trans_time"]).dt.hour

    if "trans_hour" not in df.columns:
        return go.Figure()

    hourly = df.groupby("trans_hour")["is_fraud"].agg(["count", "sum"]).reset_index()
    hourly.columns = ["hour", "total", "fraud_count"]

    # Fill all 24 hours
    all_hours = pd.DataFrame({"hour": range(24)})
    hourly = pd.merge(all_hours, hourly, on="hour", how="left").fillna(0)

    colors = [COLOR_FRAUD if (h >= 22 or h <= 5) else COLOR_PRIMARY for h in hourly["hour"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hourly["hour"],
        y=hourly["fraud_count"],
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.1)", width=1)),
        hovertemplate="<b>Hour %{x}:00</b><br>Fraud Volume: %{y}<br>Total Volume: %{customdata:,}<extra></extra>",
        customdata=hourly["total"].astype(int)
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Hour of Day (24-Hour Military Time)",
        yaxis_title="Fraud Incident Count",
        margin=dict(l=15, r=15, t=15, b=15),
        xaxis=dict(tickmode="linear", tick0=0, dtick=2, gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b"),
        font=dict(color="#94a3b8")
    )
    return fig


def create_interactive_roc_curve(roc_data: Dict[str, Any], current_threshold: Optional[float] = None) -> go.Figure:
    """Plotly interactive ROC Curve with operating point marker."""
    fpr = roc_data["fpr"]
    tpr = roc_data["tpr"]
    roc_auc = roc_data.get("roc_auc", 0.0)

    fig = go.Figure()
    # Shaded AUC area
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=f"Decision Tree (AUC = {roc_auc:.4f})",
        line=dict(color=COLOR_PRIMARY, width=3),
        fill="tozeroy",
        fillcolor="rgba(56, 189, 248, 0.12)",
        hovertemplate="False Positive Rate: %{x:.4f}<br>True Positive Rate: %{y:.4f}<extra></extra>"
    ))
    # Reference Diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random Classifier (AUC = 0.50)",
        line=dict(color="#64748b", dash="dash", width=1.5),
        hoverinfo="skip"
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Sensitivity / Recall)",
        margin=dict(l=15, r=15, t=15, b=15),
        legend=dict(x=0.52, y=0.1, bgcolor="rgba(15,23,42,0.8)", font=dict(color="#94a3b8")),
        xaxis=dict(gridcolor="#1e293b", range=[-0.02, 1.02]),
        yaxis=dict(gridcolor="#1e293b", range=[-0.02, 1.05]),
        font=dict(color="#94a3b8")
    )
    return fig


def create_interactive_confusion_matrix(metrics: Dict[str, Any]) -> go.Figure:
    """Plotly interactive Confusion Matrix heatmap."""
    tn = metrics.get("true_negatives", 0)
    fp = metrics.get("false_positives", 0)
    fn = metrics.get("false_negatives", 0)
    tp = metrics.get("true_positives", 0)

    total = tn + fp + fn + tp
    z = [[tn, fp], [fn, tp]]
    x = ["Predicted Legit (0)", "Predicted Fraud (1)"]
    y = ["Actual Legit (0)", "Actual Fraud (1)"]

    annotations = []
    text_labels = [
        [f"<b>TN</b><br>{tn:,}<br><span style='font-size:11px;color:#94a3b8'>{(tn/total)*100:.1f}%</span>",
         f"<b>FP</b><br>{fp:,}<br><span style='font-size:11px;color:#f87171'>{(fp/total)*100:.1f}%</span>"],
        [f"<b>FN</b><br>{fn:,}<br><span style='font-size:11px;color:#f87171'>{(fn/total)*100:.1f}%</span>",
         f"<b>TP</b><br>{tp:,}<br><span style='font-size:11px;color:#34d399'>{(tp/total)*100:.1f}%</span>"]
    ]

    for i in range(2):
        for j in range(2):
            annotations.append(dict(
                x=x[j], y=y[i],
                text=text_labels[i][j],
                showarrow=False,
                font=dict(size=14, color="white")
            ))

    fig = go.Figure(data=go.Heatmap(
        z=z, x=x, y=y,
        colorscale=[[0, "#0f172a"], [0.5, "#1e3a8a"], [1.0, "#0284c7"]],
        showscale=False
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=annotations,
        margin=dict(l=15, r=15, t=15, b=15),
        font=dict(color="#94a3b8")
    )
    return fig


def create_interactive_feature_importance(fi_df: pd.DataFrame) -> go.Figure:
    """Plotly interactive horizontal bar chart for feature importances."""
    top_fi = fi_df.head(10).sort_values(by="importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=top_fi["importance"],
        y=top_fi["feature"],
        orientation="h",
        marker=dict(
            color=top_fi["importance"],
            colorscale=[[0, "#0284c7"], [0.5, "#38bdf8"], [1.0, "#f43f5e"]],
            line=dict(color="rgba(255,255,255,0.1)", width=1)
        ),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Gini Feature Importance Score",
        margin=dict(l=15, r=15, t=15, b=15),
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b"),
        font=dict(color="#94a3b8")
    )
    return fig


def create_interactive_risk_gauge(prob: float) -> go.Figure:
    """Interactive Speedometer / Gauge for Real-time Fraud Probability."""
    pct = prob * 100.0
    bar_color = COLOR_FRAUD if pct >= 70 else (COLOR_WARNING if pct >= 35 else COLOR_LEGIT)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "<b>FRAUD PROBABILITY RISK GAUGE</b>", 'font': {'size': 14, 'color': '#94a3b8'}},
        number={'suffix': "%", 'font': {'size': 38, 'color': '#ffffff', 'family': 'Arial Black'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748b", 'nticks': 6},
            'bar': {'color': bar_color, 'thickness': 0.28},
            'bgcolor': "#111827",
            'borderwidth': 1,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.2)'},
                {'range': [35, 70], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.25)'}
            ],
            'threshold': {
                'line': {'color': "#ef4444", 'width': 3},
                'thickness': 0.8,
                'value': 70
            }
        }
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=240,
        margin=dict(l=20, r=20, t=35, b=15)
    )
    return fig


def create_hour_day_heatmap(df: pd.DataFrame) -> go.Figure:
    """Creates a 2D density heatmap of transaction counts across Day of Week and Hour of Day."""
    if "trans_date_trans_time" in df.columns:
        dt = pd.to_datetime(df["trans_date_trans_time"])
        hour = dt.dt.hour
        dow = dt.dt.day_name()
    elif "trans_hour" in df.columns and "trans_dayofweek" in df.columns:
        hour = df["trans_hour"]
        days = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday", 7: "Saturday"}
        dow = df["trans_dayofweek"].map(days)
    else:
        return go.Figure()

    fraud_df = df[df["is_fraud"] == 1]
    if fraud_df.empty:
        fraud_df = df

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pd.crosstab(
        pd.Categorical(dow[fraud_df.index], categories=day_order, ordered=True),
        hour[fraud_df.index]
    ).reindex(columns=range(24), fill_value=0)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in range(24)],
        y=pivot.index,
        colorscale="Reds",
        hovertemplate="Day: %{y}<br>Time: %{x}<br>Fraud Incidents: %{z}<extra></extra>"
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        margin=dict(l=15, r=15, t=15, b=15),
        font=dict(color="#94a3b8")
    )
    return fig


def create_geospatial_fraud_map(df: pd.DataFrame) -> go.Figure:
    """Plotly offline scatter geo map of USA customer transactions."""
    if not all(c in df.columns for c in ["lat", "long", "is_fraud"]):
        return go.Figure()

    # Sample for performance if large
    sample_df = df.copy()
    if len(sample_df) > 1000:
        legit_samp = sample_df[sample_df["is_fraud"] == 0].sample(n=min(900, len(sample_df[sample_df["is_fraud"] == 0])), random_state=42)
        fraud_samp = sample_df[sample_df["is_fraud"] == 1]
        sample_df = pd.concat([legit_samp, fraud_samp]).sample(frac=1.0, random_state=42)

    sample_df["Class_Label"] = sample_df["is_fraud"].map({0: "Legitimate", 1: "Fraudulent"})

    fig = px.scatter_geo(
        sample_df,
        lat="lat",
        lon="long",
        color="Class_Label",
        color_discrete_map={"Legitimate": "rgba(56, 189, 248, 0.4)", "Fraudulent": "#ef4444"},
        size=sample_df["is_fraud"].map({0: 4, 1: 12}),
        hover_name="category" if "category" in sample_df.columns else None,
        hover_data={"amt": ":$.2f", "lat": False, "long": False, "is_fraud": False},
        scope="usa",
        template="plotly_dark"
    )

    fig.update_geos(
        bgcolor="rgba(0,0,0,0)",
        landcolor="#111827",
        subunitcolor="#334155",
        countrycolor="#334155",
        showlakes=True,
        lakecolor="#0b0f19"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(color="#94a3b8"))
    )
    return fig


def create_scatter_amount_distance(df: pd.DataFrame) -> go.Figure:
    """Scatter plot: Amount vs Distance (km) with interactive drill-down."""
    if not all(c in df.columns for c in ["amt", "distance_km", "is_fraud"]):
        return go.Figure()

    plot_df = df.copy()
    plot_df["Class"] = plot_df["is_fraud"].map({0: "Legitimate (0)", 1: "Fraudulent (1)"})

    fig = px.scatter(
        plot_df,
        x="distance_km",
        y="amt",
        color="Class",
        color_discrete_map={"Legitimate (0)": "rgba(56, 189, 248, 0.5)", "Fraudulent (1)": "#ef4444"},
        hover_data={"category": True, "amt": ":$.2f", "distance_km": ":.1f km", "Class": False},
        labels={"distance_km": "Customer-Merchant Distance (km)", "amt": "Transaction Amount ($)"}
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=15, r=15, t=15, b=15),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(color="#94a3b8")),
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b"),
        font=dict(color="#94a3b8")
    )
    return fig
