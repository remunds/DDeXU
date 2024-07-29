import matplotlib.pyplot as plt
import numpy as np

import mlflow
from tueplots import bundles

# plt.rcParams.update(bundles.iclr2024(usetex=False, column="full"))
plt.rcParams.update(bundles.iclr2024(usetex=False))
plt.rcParams.update({"lines.linewidth": 2})


# Uncertainty reflects strength of corruption well
def uncert_corrupt_plot(accuracies, uncertainties, title, mode="ll"):
    fig, ax = plt.subplots()
    ax.set_xlabel("Severity", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xticks(np.array(list(range(1, 6))))

    # Set tick locations and labels
    ax.set_xticks(np.arange(len(accuracies)))
    ax.set_xticklabels([str(i + 1) for i in range(len(accuracies))])
    ax.set_ylim([0, 1])
    ax.set_xlim([0, len(accuracies) - 1])

    # Plot accuracies with dotted lines and markers
    ax.plot(
        accuracies,
        linestyle="solid",
        marker="o",
        label="Accuracy",
        color="orange",
    )

    ax2 = ax.twinx()
    label = "Marginal LL" if mode == "ll" else "Entropy (normalized)"
    ax2.set_ylabel(label, fontsize=12)
    if mode == "ll":
        # ax2.set_ylim([-12, -4]) # cifar10
        ax2.set_ylim([-20, 10])  # svhn
    else:
        # project entropy to [0,1]
        uncertainties = (uncertainties - np.min(uncertainties)) / (
            np.max(uncertainties) - np.min(uncertainties)
        )
        ax2.set_ylim([0, 1])  # svhn

    # Plot uncertainties with dotted lines and markers
    ax2.plot(
        uncertainties,
        linestyle="dashed",
        marker="o",
        label=label,
        color="blue",
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

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    location = "upper right" if mode == "ll" else "upper left"
    ax.legend(
        handles + handles2, labels + labels2, loc=location, fontsize=10
    ).set_zorder(10)

    # plt.tight_layout()
    mlflow.log_figure(fig, f"{mode}_{title}.pdf")
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
    mlflow.log_figure(fig, f"calibration_{name}.pdf")
    plt.clf()


def calibration_plot_multi(confidences, accs, names, binning, dl_name):
    plt.clf()  # clear previous plots
    fig, ax = plt.subplots()
    # Plot the calibration curve
    for i in range(len(names)):
        ax.plot(confidences[i], accs[i], marker="o", label=names[i])
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal line for reference

    # Grid
    ax.grid(True)

    # Increase thickness of lines slightly
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["top"].set_linewidth(1.5)
    ax.spines["right"].set_linewidth(1.5)

    # Make axis labels better readable
    ax.tick_params(axis="both", which="major", labelsize=10)

    ax.set_xlabel("mean confidence", fontsize=12)
    ax.set_ylabel("observed accuracy", fontsize=12)
    ax.legend(loc="upper left", fontsize=10).set_zorder(10)

    mlflow.log_figure(fig, f"calibration_multi_{binning}_{dl_name}.pdf")
    plt.clf()


# Explaining variables vs. uncertainty/confidence
def explain_plot(
    corruptions,
    uncertainty,
    explanations,
    name,
    mode="ll",
    show_legend=True,
):
    assert len(corruptions) == explanations.shape[1]
    # plt.set_cmap("tab20")
    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, len(corruptions)))
    if show_legend and mode == "ll":
        fig, ax = plt.subplots(figsize=(10, 6))
    elif show_legend and mode == "mpe":
        fig, ax = plt.subplots(figsize=(9.5, 6))
    elif show_legend and mode == "post":
        fig, ax = plt.subplots(figsize=(9.1, 6))
    else:
        fig, ax = plt.subplots(figsize=(7, 6))

    ax.set_xticks(np.arange(len(explanations)))
    ax.set_xticklabels([str(i + 1) for i in range(len(explanations))])
    ax.set_xlabel("Severity", fontsize=12)

    label = "Marginal data log-likelihood" if mode == "ll" else "Entropy"
    ax.plot(
        uncertainty,
        label=label,
        color="red",
        linestyle="solid",
        marker="o",
    )
    ax.set_ylabel(label, fontsize=12)
    ax.tick_params(axis="y")
    from matplotlib.ticker import FormatStrFormatter

    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    ax2 = ax.twinx()
    for i in range(explanations.shape[1]):
        ax2.plot(
            explanations[:, i],
            label=corruptions[i].replace("_", " ").capitalize(),
            color=colors[i],
            linestyle="dashed",
            marker="o",
        )
    ax2.tick_params(axis="y")
    mode = (
        "Likelihood explanations"
        if mode == "ll"
        else "Most probable explanation" if mode == "mpe" else "Posterior explanations"
    )
    ax2.set_ylabel(f"{mode}", fontsize=12)
    if mode == "Most probable explanation":
        # mpe-expl
        ax2.set_ylim([0, 5])

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

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            handles + handles2,
            labels + labels2,
            loc="upper left",
            bbox_to_anchor=(1.1, 1) if "like" in mode else (1.15, 1),
            fontsize=10,
        ).set_zorder(10)

    plt.tight_layout()
    return fig


def plot_brightness_binned(bins, accs, ll_expl_binned, p_expl_binned, mpe_expl_binned):

    # plot ll, p and mpe, in one plot where x-axis is the brightness value (bins) and y-axis is the explanation value
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        bins[:-1],
        accs,
        label="Accuracy",
        marker="o",
        linestyle="--",
        color="tab:red",
    )
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xlabel("Noise corruption severity", fontsize=12)
    ax.grid(True)

    ax2 = ax.twinx()
    ax2.plot(
        bins[:-1],
        ll_expl_binned,
        label="LL explanation",
        marker="o",
        color="tab:blue",
    )
    ax2.plot(
        bins[:-1],
        p_expl_binned,
        label="Posterior explanation",
        marker="o",
        color="tab:orange",
    )
    ax2.plot(
        bins[:-1],
        mpe_expl_binned,
        label="MPE",
        marker="o",
        color="tab:green",
    )
    ax2.set_ylabel("Explanations normalized in [0,1]", fontsize=12)

    # Increase thickness of lines slightly
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["top"].set_linewidth(1.5)
    ax.spines["right"].set_linewidth(1.5)

    # Make axis labels better readable

    ax.tick_params(axis="both", which="major", labelsize=10)

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles + handles2, labels + labels2, loc="upper left", fontsize=10
    ).set_zorder(10)

    plt.tight_layout()
    return fig


def plot_uncertainty_surface(
    train_examples,
    train_labels,
    test_uncertainty,
    ax,
    cmap=None,
    plot_train=True,
    ood_examples=None,
    fig6=False,
):
    """
    adapted from here: https://www.tensorflow.org/tutorials/understanding/sngp
    Visualizes the 2D uncertainty surface.

    For simplicity, assume these objects already exist in the memory:

        test_examples: Array of test examples, shape (num_test, 2).
        train_labels: Array of train labels, shape (num_train, ).
        train_examples: Array of train examples, shape (num_train, 2).

    Arguments:
        test_uncertainty: Array of uncertainty scores, shape (num_test,).
        ax: A matplotlib Axes object that specifies a matplotlib figure.
        cmap: A matplotlib colormap object specifying the palette of the
        predictive surface.

    Returns:
        pcm: A matplotlib PathCollection object that contains the palette
        information of the uncertainty plot.
    """
    import matplotlib.colors as colors

    plt.rcParams["figure.dpi"] = 140
    if fig6:
        DEFAULT_X_RANGE = (-10, 10)
        DEFAULT_Y_RANGE = (-10, 10)
    else:
        DEFAULT_X_RANGE = (-3.5, 3.5)
        DEFAULT_Y_RANGE = (-2.5, 2.5)

    DEFAULT_CMAP = colors.ListedColormap(["#377eb8", "#ff7f00", "#4daf4a"])
    DEFAULT_NORM = colors.Normalize(
        vmin=0,
        vmax=1,
    )
    DEFAULT_N_GRID = 100
    # Normalize uncertainty for better visualization.
    test_uncertainty = test_uncertainty - np.min(test_uncertainty)
    test_uncertainty = test_uncertainty / (
        np.max(test_uncertainty) - np.min(test_uncertainty)
    )

    # Set view limits.
    ax.set_ylim(DEFAULT_Y_RANGE)
    ax.set_xlim(DEFAULT_X_RANGE)

    # Make axis labels better readable
    ax.tick_params(axis="both", which="major", labelsize=25)
    # ax.tick_params(axis="both", which="minor", labelsize=15)

    # Increase thickness of lines slightly
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["top"].set_linewidth(1.5)
    ax.spines["right"].set_linewidth(1.5)

    # Plot normalized uncertainty surface.
    pcm = ax.imshow(
        np.reshape(test_uncertainty, [DEFAULT_N_GRID, DEFAULT_N_GRID]),
        cmap=cmap,
        origin="lower",
        extent=DEFAULT_X_RANGE + DEFAULT_Y_RANGE,
        vmin=DEFAULT_NORM.vmin,
        vmax=DEFAULT_NORM.vmax,
        interpolation="bicubic",
        aspect="auto",
    )

    if plot_train:
        # Plot training data.
        ax.scatter(
            train_examples[:, 0],
            train_examples[:, 1],
            c=train_labels,
            cmap=DEFAULT_CMAP,
            alpha=0.5,
        )
        if ood_examples is not None:
            ax.scatter(ood_examples[:, 0], ood_examples[:, 1], c="red", alpha=0.1)

    return pcm
