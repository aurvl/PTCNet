import matplotlib.pyplot as plt


def plot_history(history, model=""):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle(f"Courbes d'entrainement et validation {model}")

    # 1) Loss
    axes[0].plot(epochs, history["train_loss"], label="train", color="tab:blue")
    axes[0].plot(epochs, history["val_loss"], label="val", color="tab:orange")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    # 2) Accuracy
    axes[1].plot(epochs, history["train_acc"], label="train", color="tab:blue")
    axes[1].plot(epochs, history["val_acc"], label="val", color="tab:orange")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")

    # 3) PR AUC
    axes[2].plot(epochs, history["train_pr_auc"], label="train", color="tab:blue")
    axes[2].plot(epochs, history["val_pr_auc"], label="val", color="tab:orange")
    axes[2].set_title("PR AUC")
    axes[2].set_xlabel("epoch")

    # 4) ROC AUC
    axes[3].plot(epochs, history["train_roc_auc"], label="train", color="tab:blue")
    axes[3].plot(epochs, history["val_roc_auc"], label="val", color="tab:orange")
    axes[3].set_title("ROC AUC")
    axes[3].set_xlabel("epoch")

    plt.tight_layout()
    plt.show()