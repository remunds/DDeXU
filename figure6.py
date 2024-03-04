import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import mlflow

import os


def generate_data(samples=1000):
    # 2d gaussian with torch
    mean = torch.tensor([-1.8, -1.8])
    cov = torch.tensor([[1.5, 0.0], [0.0, 1.5]])
    x = torch.distributions.MultivariateNormal(mean, cov).sample((samples,))

    mean = torch.tensor([1.8, -1.8])
    cov = torch.tensor([[1.5, 0.0], [0.0, 1.5]])
    y = torch.distributions.MultivariateNormal(mean, cov).sample((samples,))

    mean = torch.tensor([0.0, 1.8])
    cov = torch.tensor([[1.5, 0.0], [0.0, 1.5]])
    z = torch.distributions.MultivariateNormal(mean, cov).sample((samples,))

    combined = torch.cat([x, y, z], dim=0)
    target = torch.cat(
        [torch.zeros(samples), torch.ones(samples), 2 * torch.ones(samples)]
    )
    return list(zip(combined, target))


def make_testing_data():
    """Create a mesh grid in 2D space."""
    plt.rcParams["figure.dpi"] = 140
    DEFAULT_X_RANGE = (-10, 10)
    DEFAULT_Y_RANGE = (-10, 10)
    DEFAULT_N_GRID = 100
    x_range = DEFAULT_X_RANGE
    y_range = DEFAULT_Y_RANGE
    n_grid = DEFAULT_N_GRID
    # testing data (mesh grid over data space)
    x = np.linspace(x_range[0], x_range[1], n_grid).astype(np.float32)
    y = np.linspace(y_range[0], y_range[1], n_grid).astype(np.float32)
    xv, yv = np.meshgrid(x, y)
    return np.stack([xv.flatten(), yv.flatten()], axis=-1)


def plot_data(x, y, z):
    import matplotlib.pyplot as plt

    plt.scatter(x[:, 0], x[:, 1], label="x")
    plt.scatter(y[:, 0], y[:, 1], label="y")
    plt.scatter(z[:, 0], z[:, 1], label="z")
    plt.legend()
    plt.savefig("figure6.png")


def plot_uncertainty_surface(
    train_examples,
    train_labels,
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
    DEFAULT_X_RANGE = (-10, 10)
    DEFAULT_Y_RANGE = (-10, 10)
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

    return pcm


def start_figure6_run(run_name, batch_sizes, model_params, train_params, trial):
    print("starting new figure6 run: ", run_name)
    with mlflow.start_run(run_name=run_name):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mlflow.log_param("device", device)

        ckpt_dir = f"ckpts/figure6/{run_name}/"
        os.makedirs(ckpt_dir, exist_ok=True)
        mlflow.log_param("ckpt_dir", ckpt_dir)

        # log all params
        mlflow.log_params(batch_sizes)
        mlflow.log_params(model_params)
        mlflow.log_params(train_params)

        train = generate_data(1000)
        valid = generate_data(100)

        train_dl = torch.utils.data.DataLoader(
            train, batch_size=batch_sizes["resnet"], shuffle=True
        )
        valid_dl = torch.utils.data.DataLoader(
            valid, batch_size=batch_sizes["resnet"], shuffle=True
        )

        model_name = model_params["model"]
        del model_params["model"]
        if model_name == "DenseResNetSNGP":
            from ResNetSPN import DenseResNetSNGP

            model = DenseResNetSNGP(**model_params)
        elif model_name == "DenseResNetSPN":
            from ResNetSPN import DenseResNetSPN

            model = DenseResNetSPN(**model_params)
        mlflow.set_tag("model", model.__class__.__name__)
        model.to(device)

        lowest_val_loss = model.start_train(
            train_dl,
            valid_dl,
            device,
            checkpoint_dir=ckpt_dir,
            trial=trial,
            **train_params,
        )
        # before costly evaluation, make sure that the model is not completely off
        valid_acc = model.eval_acc(valid_dl, device)
        mlflow.log_metric("valid_acc", valid_acc)
        # if valid_acc < 0.5:
        #     # let optuna know that this is a bad trial
        #     return lowest_val_loss

        mlflow.pytorch.log_state_dict(model.state_dict(), "model")

        test = make_testing_data()
        test_dl = torch.utils.data.DataLoader(
            test,
            batch_size=batch_sizes["resnet"],
            shuffle=False,
        )
        train_data = torch.stack([x[0] for x in train])
        train_labels = torch.stack([x[1] for x in train])

        ll_marg = model.eval_ll_marg(None, device, test_dl, return_all=True)
        print(ll_marg.shape)
        ll_marg = ll_marg.cpu().detach().numpy()

        fig, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(train_data, train_labels, ll_marg, ax=ax)
        plt.colorbar(pcm, ax=ax)
        plt.title("NLL, SPN model")
        mlflow.log_figure(fig, "nll.png")

        fig, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(
            train_data, train_labels, ll_marg, ax=ax, plot_train=False
        )
        plt.colorbar(pcm, ax=ax)
        plt.title("NLL, SPN model")
        mlflow.log_figure(fig, "nll_notrain.png")

        entropy = model.eval_entropy(None, device, test_dl, return_all=True)
        entropy = entropy.cpu().detach().numpy()

        fig, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(train_data, train_labels, entropy, ax=ax)
        plt.colorbar(pcm, ax=ax)
        plt.title("Entropy, SPN model")
        mlflow.log_figure(fig, "entr.png")

        fig, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(
            train_data, train_labels, entropy, ax=ax, plot_train=False
        )
        plt.colorbar(pcm, ax=ax)
        plt.title("Entropy, SPN model")
        mlflow.log_figure(fig, "entr_notrain.png")

        class_probability = model.eval_highest_class_prob(
            None, device, test_dl, return_all=True
        )
        class_probability = class_probability.cpu().detach().numpy()

        fig, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(
            train_data, train_labels, class_probability, ax=ax
        )
        plt.colorbar(pcm, ax=ax)
        plt.title("Highest class prob, SPN model")
        mlflow.log_figure(fig, "class_prob.png")

        logits = model.backbone_logits(test_dl, device, return_all=True)
        probs = torch.softmax(logits, dim=1)
        aleatoric = -torch.sum(probs * torch.log(probs), axis=1).cpu().detach().numpy()
        fig, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(train_data, train_labels, aleatoric, ax=ax)
        plt.colorbar(pcm, ax=ax)
        plt.title("Aleatoric, SPN model")
        mlflow.log_figure(fig, "aleatoric.png")
