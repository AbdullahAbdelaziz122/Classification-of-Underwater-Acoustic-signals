import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def parse_training_log(log_path: str):
    """Parse the training log to extract epoch metrics and final class metrics."""
    epochs_data = []
    class_data = []

    # Regex patterns matching your specific log output
    epoch_pattern = re.compile(
        r"Epoch (\d+)/\d+ \| train_loss=([\d.]+) \| train_acc=([\d.]+) \| val_loss=([\d.]+) \| val_acc=([\d.]+) \| lr=([\d.]+)"
    )
    class_pattern = re.compile(
        r"Class=(\w+) \| accuracy=([\d.]+) \| precision=([\d.]+) \| recall=([\d.]+) \| f1=([\d.]+) \| support=(\d+)"
    )

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                epochs_data.append(
                    {
                        "Epoch": int(epoch_match.group(1)),
                        "Train Loss": float(epoch_match.group(2)),
                        "Train Acc": float(epoch_match.group(3)),
                        "Val Loss": float(epoch_match.group(4)),
                        "Val Acc": float(epoch_match.group(5)),
                        "Learning Rate": float(epoch_match.group(6)),
                    }
                )

            class_match = class_pattern.search(line)
            if class_match:
                class_data.append(
                    {
                        "Class": class_match.group(1),
                        "Accuracy": float(class_match.group(2)),
                        "Precision": float(class_match.group(3)),
                        "Recall": float(class_match.group(4)),
                        "F1-Score": float(class_match.group(5)),
                        "Support": int(class_match.group(6)),
                    }
                )

    return pd.DataFrame(epochs_data), pd.DataFrame(class_data)


def plot_learning_curves(df_epochs: pd.DataFrame, output_dir: Path):
    """Generate and save loss, accuracy, and learning rate curves."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss Curve
    sns.lineplot(
        x="Epoch",
        y="Train Loss",
        data=df_epochs,
        ax=axes[0],
        label="Train",
        linewidth=2,
    )
    sns.lineplot(
        x="Epoch",
        y="Val Loss",
        data=df_epochs,
        ax=axes[0],
        label="Validation",
        linewidth=2,
    )
    axes[0].set_title("Training and Validation Loss", fontweight="bold")
    axes[0].set_ylabel("Cross-Entropy Loss")

    # Accuracy Curve
    sns.lineplot(
        x="Epoch", y="Train Acc", data=df_epochs, ax=axes[1], label="Train", linewidth=2
    )
    sns.lineplot(
        x="Epoch",
        y="Val Acc",
        data=df_epochs,
        ax=axes[1],
        label="Validation",
        linewidth=2,
    )
    axes[1].set_title("Training and Validation Accuracy", fontweight="bold")
    axes[1].set_ylabel("Accuracy")

    # Learning Rate Curve
    sns.lineplot(
        x="Epoch",
        y="Learning Rate",
        data=df_epochs,
        ax=axes[2],
        color="purple",
        linewidth=2,
    )
    axes[2].set_title("Learning Rate Schedule", fontweight="bold")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_dir / "learning_curves.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_class_metrics(df_class: pd.DataFrame, output_dir: Path):
    """Generate and save a bar chart for final per-class metrics."""
    if df_class.empty:
        return

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # Melt the dataframe for seaborn grouped barplot
    df_melted = df_class.melt(
        id_vars=["Class"],
        value_vars=["Precision", "Recall", "F1-Score"],
        var_name="Metric",
        value_name="Score",
    )

    plt.figure(figsize=(14, 6))
    sns.barplot(x="Class", y="Score", hue="Metric", data=df_melted, palette="viridis")

    plt.title("Per-Class Evaluation Metrics", fontweight="bold")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.xlabel("Acoustic Class")
    plt.legend(loc="lower right")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_dir / "per_class_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    log_file = (
        "../../outputs/training_log.txt"  # Update this if your log is named differently
    )
    out_dir = Path("../../outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing {log_file}...")
    df_epochs, df_class = parse_training_log(log_file)

    if not df_epochs.empty:
        plot_learning_curves(df_epochs, out_dir)
        print(f"Saved learning curves to {out_dir / 'learning_curves.png'}")

    if not df_class.empty:
        plot_class_metrics(df_class, out_dir)
        print(f"Saved class metrics to {out_dir / 'per_class_metrics.png'}")
