import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script rendering
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set aesthetic style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

def plot_fraud_distribution_by_category(df_pandas, save_path=None):
    """
    Visualizes fraud distribution by category:
    1. Total transactions and fraud count per category.
    2. Fraud rate (%) per category.
    """
    cat_summary = (
        df_pandas.groupby("category")
        .agg(
            total_trans=("is_fraud", "count"),
            fraud_trans=("is_fraud", "sum"),
            fraud_rate=("is_fraud", lambda x: (x.sum() / len(x)) * 100),
            avg_amt=("amt", "mean"),
            total_amt=("amt", "sum")
        )
        .sort_values(by="fraud_trans", ascending=False)
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Total vs Fraud Count
    bars = sns.barplot(
        data=cat_summary,
        x="category",
        y="fraud_trans",
        hue="category",
        legend=False,
        ax=axes[0],
        palette="Reds_r"
    )
    axes[0].set_title("Fraudulent Transaction Count by Category", fontsize=14, fontweight="bold", pad=12)
    axes[0].set_xlabel("Merchant Category", fontsize=12)
    axes[0].set_ylabel("Number of Fraud Cases", fontsize=12)
    axes[0].tick_params(axis="x", rotation=45)
    
    # Add value labels on bars
    for p in axes[0].patches:
        height = p.get_height()
        if height > 0:
            axes[0].annotate(f"{int(height)}",
                xy=(p.get_x() + p.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 2. Fraud Rate (%)
    rate_df = cat_summary.sort_values(by="fraud_rate", ascending=False)
    sns.barplot(
        data=rate_df,
        x="category",
        y="fraud_rate",
        hue="category",
        legend=False,
        ax=axes[1],
        palette="Blues_r"
    )
    axes[1].set_title("Fraud Incident Rate (%) by Category", fontsize=14, fontweight="bold", pad=12)
    axes[1].set_xlabel("Merchant Category", fontsize=12)
    axes[1].set_ylabel("Fraud Rate (%)", fontsize=12)
    axes[1].tick_params(axis="x", rotation=45)

    for p in axes[1].patches:
        height = p.get_height()
        if height > 0:
            axes[1].annotate(f"{height:.2f}%",
                xy=(p.get_x() + p.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, cat_summary

def plot_auc_roc_curve(roc_data, save_path=None):
    """
    Plots the Receiver Operating Characteristic (ROC) curve with AUC.
    """
    fpr = roc_data["fpr"]
    tpr = roc_data["tpr"]
    auc_score = roc_data["auc"]

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color="#d9534f", lw=2.5, label=f"Decision Tree (AUC = {auc_score:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#d9534f")
    
    # Diagonal reference line (Random chance)
    ax.plot([0, 1], [0, 1], color="#6c757d", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.5000)")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, labelpad=8)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12, labelpad=8)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9, fontsize=11)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig

def plot_confusion_matrix(cm_dict, save_path=None):
    """
    Plots a Confusion Matrix heatmap.
    cm_dict = {'tn': int, 'fp': int, 'fn': int, 'tp': int}
    """
    cm_matrix = np.array([
        [cm_dict["tn"], cm_dict["fp"]],
        [cm_dict["fn"], cm_dict["tp"]]
    ])
    
    total = np.sum(cm_matrix)
    labels = [
        [f"True Neg\n{cm_matrix[0,0]}\n({cm_matrix[0,0]/total*100:.1f}%)", f"False Pos\n{cm_matrix[0,1]}\n({cm_matrix[0,1]/total*100:.1f}%)"],
        [f"False Neg\n{cm_matrix[1,0]}\n({cm_matrix[1,0]/total*100:.1f}%)", f"True Pos\n{cm_matrix[1,1]}\n({cm_matrix[1,1]/total*100:.1f}%)"]
    ]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_matrix,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        annot_kws={"fontsize": 11, "fontweight": "bold"},
        ax=ax
    )
    ax.set_title("Decision Tree Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xticklabels(["Legitimate (0)", "Fraudulent (1)"], fontsize=11)
    ax.set_yticklabels(["Legitimate (0)", "Fraudulent (1)"], fontsize=11)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig

def plot_feature_importances(fi_df, save_path=None):
    """
    Plots horizontal bar chart of Decision Tree feature importances.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=fi_df,
        y="feature",
        x="importance_pct",
        hue="feature",
        legend=False,
        palette="viridis",
        ax=ax
    )
    ax.set_title("Decision Tree Feature Importance (%)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Importance Percentage (%)", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    
    for p in ax.patches:
        width = p.get_width()
        if width > 0.1:
            ax.annotate(f"{width:.1f}%",
                xy=(width, p.get_y() + p.get_height() / 2),
                xytext=(5, 0), textcoords="offset points",
                ha="left", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig

def plot_spending_behavior(df_pandas, save_path=None):
    """
    Visualizes spending behavior:
    1. Transaction amount boxplot comparison (Log Scale) between Normal vs Fraud.
    2. Hourly transaction distribution for Fraud vs Normal.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # 1. Amount Boxplot
    df_plot = df_pandas.copy()
    df_plot["Fraud_Label"] = df_plot["is_fraud"].map({0: "Legitimate (0)", 1: "Fraud (1)"})
    
    sns.boxplot(
        data=df_plot,
        x="Fraud_Label",
        y="amt",
        hue="Fraud_Label",
        legend=False,
        palette=["#2b8a3e", "#c92a2a"],
        ax=axes[0],
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"}
    )
    axes[0].set_yscale("log")
    axes[0].set_title("Spending Amount Distribution (Log Scale)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Transaction Class", fontsize=11)
    axes[0].set_ylabel("Transaction Amount ($ - Log)", fontsize=11)

    # 2. Hourly Fraud Pattern
    if "trans_hour" not in df_plot.columns and "trans_date_trans_time" in df_plot.columns:
        df_plot["trans_hour"] = pd.to_datetime(df_plot["trans_date_trans_time"]).dt.hour

    hourly = df_plot.groupby(["trans_hour", "Fraud_Label"]).size().unstack(fill_value=0)
    if "Fraud (1)" in hourly.columns:
        hourly["Fraud (1)"].plot(kind="bar", color="#c92a2a", ax=axes[1], width=0.7)
        axes[1].set_title("Fraud Frequency by Hour of Day (0-23)", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Hour of Day (24-Hour)", fontsize=11)
        axes[1].set_ylabel("Number of Fraud Cases", fontsize=11)
        axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
