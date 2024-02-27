import matplotlib.pyplot as plt
import numpy as np
import mlflow


# Uncertainty reflects strength of corruption well
def uncert_corrupt_plot(accuracies, uncertainties, title, mode="ll"):
    fig, ax = plt.subplots()
    ax.set_xlabel("severity")
    ax.set_xticks(np.array(list(range(1, 6))))

    ax.plot(accuracies, label="accuracy", color="red")
    ax.set_ylabel("accuracy")
    ax.tick_params(axis="y")
    ax.set_ylim([0, 1])

    ax2 = ax.twinx()
    label = "marginal ll" if mode == "ll" else "entropy"
    ax2.plot(uncertainties, label=label, color="blue")
    ax2.tick_params(axis="y")
    ax2.set_ylabel(label)

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    location = "upper right" if mode == "ll" else "upper left"
    ax.legend(
        handles + handles2,
        labels + labels2,
        loc=location,
        # loc="upper left",
        # bbox_to_anchor=(1, 1),
    ).set_zorder(10)
    plt.tight_layout()
    mlflow.log_figure(fig, f"{mode}_{title}.png")
    plt.clf()


def histogram_plot(confidences, n_bins, title):
    # show histogram of confidences
    fig, ax = plt.subplots()
    ax.set_xlabel("confidence")
    ax.set_ylabel("frequency")
    ax.hist(confidences, bins=n_bins)
    mlflow.log_figure(fig, f"hist_{title}.png")
    plt.clf()


# Uncertainties are well calibrated
def calibration_plot(confidences, accs, ece, nll, name):
    fig, ax = plt.subplots()
    # Plot the calibration curve
    ax.plot(confidences, accs, marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal line for reference
    ax.set_xlabel("mean confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_title(
        "calibration plot, ECE: {ece:.3f}, NLL: {nll:.3f}".format(ece=ece, nll=nll)
    )
    mlflow.log_figure(fig, f"calibration_{name}.png")
    plt.clf()


# Explaining variables vs. uncertainty/confidence
def explain_plot(
    corruptions,
    uncertainty,
    explanations,
    name,
    mode="ll",
):
    assert len(corruptions) == explanations.shape[1]
    # plt.set_cmap("tab20")
    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, len(corruptions)))

    fig, ax = plt.subplots()
    ax.set_xlabel("severity")
    ax.set_xticks(np.array(list(range(1, 6))))

    label = "marginal ll" if mode == "ll" else "entropy"
    ax.plot(uncertainty, label=label, color="red")
    ax.set_ylabel(label)
    ax.tick_params(axis="y")

    ax2 = ax.twinx()
    for i in range(explanations.shape[1]):
        ax2.plot(explanations[:, i], label=corruptions[i], color=colors[i])
    ax2.tick_params(axis="y")
    ax2.set_ylabel(f"{mode} explanations")
    if mode == "ll":
        # ll-expl
        ax2.set_ylim([0, 100])
    if mode == "mpe":
        # mpe-expl
        ax2.set_ylim([0, 5])

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    # location = "upper right" if mode == "ll" else "upper left"
    ax.legend(
        handles + handles2,
        labels + labels2,
        # loc=location,
        loc="upper left",
        # bbox_to_anchor=(1, 1),
    ).set_zorder(10)
    plt.tight_layout()
    mlflow.log_figure(fig, f"{name}_expl_{mode}.png")
    plt.close()
