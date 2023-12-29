import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sklearn.datasets

from torch.utils.data import DataLoader
import os
import mlflow
import optuna

### data and stuff from here: https://www.tensorflow.org/tutorials/understanding/sngp
### visualization macros
plt.rcParams["figure.dpi"] = 140

DEFAULT_X_RANGE = (-3.5, 3.5)
DEFAULT_Y_RANGE = (-2.5, 2.5)
DEFAULT_CMAP = colors.ListedColormap(["#377eb8", "#ff7f00"])
DEFAULT_NORM = colors.Normalize(
    vmin=0,
    vmax=1,
)
DEFAULT_N_GRID = 100


def plot_uncertainty_surface(
    train_examples,
    train_labels,
    ood_examples,
    test_uncertainty,
    ax,
    cmap=None,
    plot_train=True,
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
    # Normalize uncertainty for better visualization.
    # 1. all positive
    # if np.min(test_uncertainty) > 0:
    #     test_uncertainty = test_uncertainty / np.max(test_uncertainty)
    # # 2. all negative or mixed
    # elif np.max(test_uncertainty) < 0:
    #     # shift to positive
    test_uncertainty = test_uncertainty - np.min(test_uncertainty)
    test_uncertainty = test_uncertainty / np.max(test_uncertainty)

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


def make_testing_data(
    x_range=DEFAULT_X_RANGE, y_range=DEFAULT_Y_RANGE, n_grid=DEFAULT_N_GRID
):
    """Create a mesh grid in 2D space."""
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

        from ResNetSPN import DenseResNetSPN

        resnet = DenseResNetSPN(**model_params)
        mlflow.set_tag("model", resnet.__class__.__name__)
        print(resnet)
        resnet.to(device)
        # it is interesting to play with lambda_v, dropout, repetition and depth
        lowest_val_loss = resnet.start_train(
            train_dl,
            valid_dl,
            device,
            checkpoint_dir=ckpt_dir,
            trial=trial,
            **train_params,
        )
        trial.report(lowest_val_loss, 1)
        if trial.should_prune():
            raise optuna.TrialPruned()

        mlflow.pytorch.log_model(resnet, "resnet_spn")
        # evaluate
        resnet.eval()
        valid_acc = resnet.eval_acc(valid_dl, device)
        print("valid accuracy: ", valid_acc)
        mlflow.log_metric("valid accuracy", valid_acc)

        # get LL's
        pos_ll = resnet(torch.from_numpy(pos_examples).to(device)).mean()
        mlflow.log_metric("pos_ll", pos_ll)
        neg_ll = resnet(torch.from_numpy(neg_examples).to(device)).mean()
        mlflow.log_metric("neg_ll", neg_ll)

        # get LL's
        ood_ll = resnet(torch.from_numpy(ood_examples).to(device)).mean()
        mlflow.log_metric("ood_ll", ood_ll)

        resnet_logits = resnet(torch.from_numpy(test_examples).to(device))
        resnet_probs = torch.nn.functional.softmax(resnet_logits, dim=1)[:, 0]
        resnet_probs = resnet_probs.cpu().detach().numpy()
        fig, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(
            train_examples, train_labels, ood_examples, resnet_probs, ax=ax
        )

        plt.colorbar(pcm, ax=ax)
        plt.title("Class Probability, SPN model")
        mlflow.log_figure(fig, "class_probability.png")

        resnet_uncertainty = resnet_probs * (1 - resnet_probs)  # predictive variance
        fig, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(
            train_examples, train_labels, ood_examples, resnet_uncertainty, ax=ax
        )
        plt.colorbar(pcm, ax=ax)
        plt.title("Predictive Uncertainty, SPN Model")
        mlflow.log_figure(fig, "predictive_variance.png")

        resnet_uncertainty = (
            resnet_logits.cpu().detach().numpy()[:, 0]
        )  # log likelihood
        print(resnet_uncertainty[:5])
        fig, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(
            train_examples, train_labels, ood_examples, resnet_uncertainty, ax=ax
        )
        plt.colorbar(pcm, ax=ax)
        plt.title("LL Uncertainty, SPN Model")
        mlflow.log_figure(fig, "ll_uncertainty.png")

        fig, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(
            train_examples,
            train_labels,
            ood_examples,
            resnet_uncertainty,
            ax=ax,
            plot_train=False,
        )
        plt.colorbar(pcm, ax=ax)
        plt.title("LL Uncertainty, SPN Model")
        mlflow.log_figure(fig, "ll_uncertainty_no_train.png")

        test_dl = DataLoader(
            test_examples,
            batch_size=batch_sizes["resnet"],
            pin_memory=True,
            num_workers=1,
        )
        resnet_uncertainty = (
            resnet.eval_dempster_shafer(test_dl, device, return_all=True)
            .cpu()
            .detach()
            .numpy()
        )
        fig, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(
            train_examples, train_labels, ood_examples, resnet_uncertainty, ax=ax
        )
        plt.colorbar(pcm, ax=ax)
        plt.title("Dempster Shafer, SPN Model")
        mlflow.log_figure(fig, "dempster_shafer.png")

        # return train_acc
        return lowest_val_loss
