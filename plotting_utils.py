import matplotlib.pyplot as plt
import numpy as np
import mlflow


# Uncertainty reflects strength of corruption well
def uncert_corrupt_plot(accuracies, uncertainties, title, mode="ll"):
    fig, ax = plt.subplots()
    ax.set_xlabel("Severity", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xticks(np.array(list(range(1, 6))))

    # # Adjust x-axis limits to start at 1
    # ax.set_xlim([1, len(accuracies)])

    # # Add a minor tick at 0
    # ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    # Set tick locations and labels
    ax.set_xticks(np.arange(len(accuracies)))
    ax.set_xticklabels([str(i + 1) for i in range(len(accuracies))])
    ax.set_ylim([0, 1])

    # Plot accuracies with dotted lines and markers
    ax.plot(
        accuracies,
        linestyle="solid",
        marker="o",
        label="Accuracy",
        color="orange",
        linewidth=1.5,
    )

    ax2 = ax.twinx()
    label = "Marginal LL" if mode == "ll" else "Entropy"
    ax2.set_ylabel(label, fontsize=12)
    ax2.set_ylim([-12, -4])

    # Plot uncertainties with dotted lines and markers
    ax2.plot(
        uncertainties,
        linestyle="dashed",
        marker="o",
        label=label,
        color="blue",
        linewidth=1.5,
    )

    # Grid
    ax.grid(True)

    # Increase thickness of lines slightly
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["top"].set_linewidth(1.5)
    ax.spines["right"].set_linewidth(1.5)
    ax2.spines["bottom"].set_linewidth(1.5)
    ax2.spines["left"].set_linewidth(1.5)
    ax2.spines["top"].set_linewidth(1.5)
    ax2.spines["right"].set_linewidth(1.5)

    # Make axis labels better readable
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax2.tick_params(axis="both", which="major", labelsize=10)

    ax.set_title(title, fontsize=14)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    location = "upper right" if mode == "ll" else "upper left"
    ax.legend(
        handles + handles2, labels + labels2, loc=location, fontsize=10
    ).set_zorder(10)

    plt.tight_layout()
    plt.show()
    # mlflow.log_figure(fig, f"{mode}_{title}.pdf")
    return fig


def histogram_plot(confidences, n_bins, title):
    # show histogram of confidences
    fig, ax = plt.subplots()
    ax.set_xlabel("confidence")
    ax.set_ylabel("fraction of data")
    ax.hist(confidences, bins=n_bins, density=True)
    mlflow.log_figure(fig, f"hist_{title}.pdf")
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
    mlflow.log_figure(fig, f"calibration_{name}.pdf")
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

    fig, ax = plt.subplots(figsize=(10, 6))
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
    if mode == "mpe":
        # mpe-expl
        ax2.set_ylim([0, 5])

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles + handles2,
        labels + labels2,
        loc="upper left",
        bbox_to_anchor=(1.1, 1),
    ).set_zorder(10)
    plt.tight_layout()
    mlflow.log_figure(fig, f"{name}_expl_{mode}.pdf")
    plt.close()
