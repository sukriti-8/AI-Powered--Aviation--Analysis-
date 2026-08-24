import os

import matplotlib.pyplot as plt
import seaborn as sns


# Create folder for saved graphs
def create_graph_folder(path="reports/figures"):

    os.makedirs(
        path,
        exist_ok=True
    )


# ============================================================
# ALL 3 PROJECT GRAPHS
# ============================================================

def show_project_visualizations(
    train,
    history,
    actual_rul,
    predictions,
    save_path="reports/figures/project_visualizations.png"
):

    create_graph_folder()

    # Create one figure containing 3 graphs
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(21, 6)
    )


    # ========================================================
    # GRAPH 1 — RUL DISTRIBUTION
    # ========================================================

    sns.histplot(
        data=train,
        x="RUL",
        bins=26,
        kde=True,
        ax=axes[0]
    )

    axes[0].set_title(
        "Training RUL Distribution"
    )

    axes[0].set_xlabel(
        "Remaining Useful Life (cycles)"
    )

    axes[0].set_ylabel(
        "Number of observations"
    )


    # ========================================================
    # GRAPH 2 — LSTM TRAINING VS VALIDATION LOSS
    # ========================================================

    epochs = range(
        1,
        len(history.history["loss"]) + 1
    )

    sns.lineplot(
        x=epochs,
        y=history.history["loss"],
        marker="o",
        label="Training Loss",
        ax=axes[1]
    )

    sns.lineplot(
        x=epochs,
        y=history.history["val_loss"],
        marker="o",
        label="Validation Loss",
        ax=axes[1]
    )

    # Find best validation epoch
    best_epoch = (
        history.history["val_loss"].index(
            min(history.history["val_loss"])
        ) + 1
    )

    axes[1].axvline(
        best_epoch,
        linestyle="--",
        label=f"Best epoch = {best_epoch}"
    )

    axes[1].set_title(
        "LSTM Training vs Validation Loss"
    )

    axes[1].set_xlabel(
        "Epoch"
    )

    axes[1].set_ylabel(
        "MSE Loss"
    )

    axes[1].legend()


    # ========================================================
    # GRAPH 3 — ACTUAL VS PREDICTED RUL
    # ========================================================

    sns.scatterplot(
        x=actual_rul,
        y=predictions,
        s=60,
        ax=axes[2]
    )

    # Perfect prediction reference line
    minimum = min(
        actual_rul.min(),
        predictions.min()
    )

    maximum = max(
        actual_rul.max(),
        predictions.max()
    )

    axes[2].plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        label="Perfect Prediction"
    )

    axes[2].set_title(
        "LSTM: Actual vs Predicted RUL"
    )

    axes[2].set_xlabel(
        "Actual RUL (cycles)"
    )

    axes[2].set_ylabel(
        "Predicted RUL (cycles)"
    )

    axes[2].legend()


    # ========================================================
    # FINAL LAYOUT
    # ========================================================

    fig.suptitle(
        "AI-Powered Aviation Predictive Maintenance",
        fontsize=16
    )

    plt.tight_layout()

    # Save all three graphs together
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    # Show all three at once
    plt.show()

    plt.close()