import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from labeling.triple_barrier import TripleBarrierConfig


def plot_labels(labels: pd.Series, config: TripleBarrierConfig) -> None:
    label_counts = labels.value_counts().sort_index()

    # Create a mapping for human-readable labels
    label_map = {
        1: f"Long (PT = {config.barrier_multiplier}x Spread)",
        0: f"Short (SL = {config.barrier_multiplier}x Spread)",
        2: f"Timeout ({config.timeout_sec}s)",
    }
    label_counts.rename(index=label_map, inplace=True)

    print(f"Label Distribution Summary (excluding last {config.timeout_sec}s of data):")
    print(f"Total samples analyzed: {len(labels):,}")
    summary_df = pd.DataFrame(
        {
            "count": label_counts,
            "percentage": (label_counts / label_counts.sum() * 100).round(2),
        }
    )
    print(summary_df)

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=summary_df.index, y=summary_df["count"])
    ax.set_title("Triple-Barrier Label Distribution")
    ax.set_ylabel("Count")
    ax.set_xlabel("Outcome")

    # Add percentage labels to the bars
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width() / 2.0,
            height + 0.1,
            f"{summary_df['percentage'].iloc[i]}%",
            ha="center",
            va="bottom",
        )

    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()
