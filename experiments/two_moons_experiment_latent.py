import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sklearn.datasets

from torch.utils.data import DataLoader
import os
import mlflow

### data and stuff from here: https://www.tensorflow.org/tutorials/understanding/sngp
### visualization macros


def plot_uncertainty_surface(
    train_examples,
    train_labels,
    ood_examples,
    test_uncertainty,
    ax,
    cmap=None,
    plot_train=True,
    input_is_ll=False,
):
    """Visualizes the 2D uncertainty surface.

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
    plt.rcParams["figure.dpi"] = 140
    DEFAULT_X_RANGE = (-3.5, 3.5)
    DEFAULT_Y_RANGE = (-2.5, 2.5)
    DEFAULT_CMAP = colors.ListedColormap(["#377eb8", "#ff7f00"])
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
        ax.scatter(ood_examples[:, 0], ood_examples[:, 1], c="red", alpha=0.1)

    return pcm


def make_training_data(sample_size=500):
    """Create two moon training dataset."""
    train_examples, train_labels = sklearn.datasets.make_moons(
        n_samples=2 * sample_size, noise=0.1
    )

    # Adjust data position slightly.
    train_examples[train_labels == 0] += [-0.1, 0.2]
    train_examples[train_labels == 1] += [0.1, -0.2]

    return train_examples.astype(np.float32), train_labels.astype(np.int32)


def make_testing_data():
    """Create a mesh grid in 2D space."""
    plt.rcParams["figure.dpi"] = 140
    DEFAULT_X_RANGE = (-3.5, 3.5)
    DEFAULT_Y_RANGE = (-2.5, 2.5)
    DEFAULT_N_GRID = 100
    x_range = DEFAULT_X_RANGE
    y_range = DEFAULT_Y_RANGE
    n_grid = DEFAULT_N_GRID
    # testing data (mesh grid over data space)
    x = np.linspace(x_range[0], x_range[1], n_grid).astype(np.float32)
    y = np.linspace(y_range[0], y_range[1], n_grid).astype(np.float32)
    xv, yv = np.meshgrid(x, y)
    return np.stack([xv.flatten(), yv.flatten()], axis=-1)


def make_ood_data(sample_size=500, means=(2.5, -1.75), vars=(0.01, 0.01)):
    return np.random.multivariate_normal(
        means, cov=np.diag(vars), size=sample_size
    ).astype(np.float32)


def start_two_moons_run(run_name, batch_sizes, model_params, train_params, trial):
    print("starting new two-moons run: ", run_name)
    with mlflow.start_run(run_name=run_name):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mlflow.log_param("device", device)

        ckpt_dir = f"ckpts/two_moons/{run_name}/"
        os.makedirs(ckpt_dir, exist_ok=True)
        mlflow.log_param("ckpt_dir", ckpt_dir)

        # log all params
        mlflow.log_params(batch_sizes)
        mlflow.log_params(model_params)
        mlflow.log_params(train_params)

        # Load the train, test and OOD datasets.
        train_examples, train_labels = make_training_data(sample_size=500)
        test_examples = make_testing_data()
        ood_examples = make_ood_data(sample_size=500)

        pos_examples = train_examples[train_labels == 0]
        neg_examples = train_examples[train_labels == 1]

        # put into data loaders
        train_ds = list(zip(train_examples, train_labels))
        train_dl = DataLoader(
            train_ds[:900],
            batch_size=batch_sizes["resnet"],
            shuffle=True,
            pin_memory=True,
            num_workers=1,
        )
        valid_dl = DataLoader(
            train_ds[900:],
            batch_size=batch_sizes["resnet"],
            shuffle=True,
            pin_memory=True,
            num_workers=1,
        )

        from gmm_utils import gmm_fit, gmm_get_logits

        # use gaussians as embeddings
        # embeddings have shape (num_samples, num_gaussians)
        gmm, jitter = gmm_fit(train_examples, train_labels, num_classes=3)
        print("jitter: ", jitter)

        test_dl = DataLoader(
            test_examples,
            batch_size=batch_sizes["resnet"],
            pin_memory=True,
            num_workers=1,
        )

        lls = gmm_get_logits(gmm, test_examples)
        print(lls.shape)
        lls = torch.logsumexp(lls, dim=1)
        nll = -(lls.cpu().detach().numpy())  # negative log likelihood

        fig, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(
            train_examples, train_labels, ood_examples, nll, ax=ax
        )
        plt.colorbar(pcm, ax=ax)
        plt.title("NLL")
        mlflow.log_figure(fig, "nll.pdf")

        fig, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(
            train_examples,
            train_labels,
            ood_examples,
            nll,
            ax=ax,
            plot_train=False,
        )
        plt.colorbar(pcm, ax=ax)
        plt.title("NLL")
        mlflow.log_figure(fig, "nll_no_train.pdf")

        return 0
