import matplotlib.pyplot as plt
import pandas as pd


def plot_loss_curve(history, title="Loss Curve"):
    if not history:
        raise ValueError("history rỗng, không thể vẽ loss curve.")

    history_df = pd.DataFrame(history)

    plt.figure(figsize=(8, 5))
    plt.plot(
        history_df["epoch"],
        history_df["train_loss"],
        label="Train Loss",
    )
    plt.plot(
        history_df["epoch"],
        history_df["valid_loss"],
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy + L2 Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()