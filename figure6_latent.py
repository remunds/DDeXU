import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import mlflow

import os

from simple_einet.einet import Einet, EinetConfig
from simple_einet.layers.distributions.normal import Normal, RatNormal


def generate_data(samples=1000):
    # # 2d gaussian with torch
    # mean = torch.tensor([-1.8, -1.8])
    # cov = torch.tensor([[1.5, 0.0], [0.0, 1.5]])

    # mean = torch.tensor([1.8, -1.8])
    # cov = torch.tensor([[1.5, 0.0], [0.0, 1.5]])
    # y = torch.distributions.MultivariateNormal(mean, cov).sample((samples,))

    # mean = torch.tensor([0.0, 1.8])
    # cov = torch.tensor([[1.5, 0.0], [0.0, 1.5]])
    # z = torch.distributions.MultivariateNormal(mean, cov).sample((samples,))

    # Define means for the Gaussians
    # mean1 = torch.tensor([-5.0, 5.0])
    # mean2 = torch.tensor([5.0, 5.0])
    # mean3 = torch.tensor([0.0, -5.0])
    mean1 = torch.tensor([-2.0, 2.0])
    mean2 = torch.tensor([2.0, 2.0])
    mean3 = torch.tensor([0.0, -2.0])

    # Define covariance matrices to stretch the Gaussians towards the middle
    covariance1 = torch.tensor([[2.0, -1.0], [-1.0, 2.0]])

    covariance2 = torch.tensor([[2.0, 1.0], [1.0, 2.0]])

    covariance3 = torch.tensor([[1.0, 0.0], [0.0, 3.0]])
    x = torch.distributions.MultivariateNormal(mean1, covariance1).sample((samples,))
    y = torch.distributions.MultivariateNormal(mean2, covariance2).sample((samples,))
    z = torch.distributions.MultivariateNormal(mean3, covariance3).sample((samples,))

    combined = torch.cat([x, y, z], dim=0)

    x_target = torch.zeros(samples)
    y_target = torch.ones(samples)
    z_target = 2 * torch.ones(samples)

    # add 4% label noise
    x_noise = torch.rand(samples) < 0.04
    y_noise = torch.rand(samples) < 0.04
    z_noise = torch.rand(samples) < 0.04

    x_target[x_noise] = torch.tensor(
        [np.random.choice([1, 2]) for _ in range(x_noise.sum())]
    ).float()
    y_target[y_noise] = torch.tensor(
        [np.random.choice([0, 2]) for _ in range(y_noise.sum())]
    ).float()
    z_target[z_noise] = torch.tensor(
        [np.random.choice([0, 1]) for _ in range(z_noise.sum())]
    ).float()

    # target = torch.cat(
    #     [torch.zeros(samples), torch.ones(samples), 2 * torch.ones(samples)]
    # )
    target = torch.cat([x_target, y_target, z_target])

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


def plot_data(data, labels):
    DEFAULT_CMAP = colors.ListedColormap(["#377eb8", "#ff7f00", "#4daf4a"])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(
        data[:, 0],
        data[:, 1],
        c=labels,
        cmap=DEFAULT_CMAP,
        alpha=0.5,
    )
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    mlflow.log_figure(fig, "figure6.png")
    plt.clf()


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
    # set seeds
    torch.manual_seed(1)
    np.random.seed(1)
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

        train = generate_data(1500)
        valid = generate_data(100)

        train_dl = torch.utils.data.DataLoader(
            train, batch_size=512, shuffle=True, num_workers=4, pin_memory=True
        )

        test = make_testing_data()
        train_data = torch.stack([x[0] for x in train])
        train_labels = torch.stack([x[1] for x in train])
        plot_data(train_data, train_labels)
        cfg = EinetConfig(
            num_features=2,
            num_channels=1,
            depth=1,
            num_sums=15,
            num_leaves=3,
            num_repetitions=15,
            num_classes=3,
            leaf_type=RatNormal,
            leaf_kwargs={
                "min_sigma": 0.000001,
                "max_sigma": 150.0,
            },
            layer_type="einsum",
            dropout=0.0,
        )
        model = Einet(cfg)

        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        lambda_v = 0.5
        divisor = 2
        from tqdm import tqdm

        t = tqdm(range(200))
        for epoch in t:
            t.set_description(f"Epoch  {epoch}")
            model.train()
            ce_loss = 0.0
            nll_loss = 0.0
            loss = 0.0
            for data, target in train_dl:
                optimizer.zero_grad()
                data = data.to(device)
                target = target.type(torch.LongTensor)
                target = target.to(device)
                output = model(data)
                ce_loss_v = lambda_v * torch.nn.CrossEntropyLoss()(output, target)
                ce_loss += ce_loss_v.item()
                nll_loss_v = (1 - lambda_v) * -(output.mean() / divisor)
                nll_loss += nll_loss_v.item()

                loss_v = ce_loss_v + nll_loss_v
                loss_v.backward()
                optimizer.step()
                loss += loss_v.item()

            t.set_postfix(
                dict(
                    train_loss=loss / len(train_dl.dataset),
                )
            )
            mlflow.log_metric(
                key="ce_loss_train",
                value=ce_loss / len(train_dl.dataset),
                step=epoch,
            )
            mlflow.log_metric(
                key="nll_loss_train",
                value=nll_loss / len(train_dl.dataset),
                step=epoch,
            )
        model.eval()

        from gmm_utils import gmm_fit, gmm_get_logits

        # use gaussians as embeddings
        # embeddings have shape (num_samples, num_gaussians)
        gmm, jitter = gmm_fit(train_data, train_labels, num_classes=3)
        print("jitter: ", jitter)
        test_tensor = torch.tensor(test)
        logits = gmm_get_logits(gmm, test_tensor)
        print(logits.shape)
        ll_marg = torch.logsumexp(logits, dim=1)
        ll_marg_cpu = ll_marg.cpu().detach().numpy()

        fig, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(train_data, train_labels, -ll_marg_cpu, ax=ax)
        plt.colorbar(pcm, ax=ax)
        plt.title("Marginal NLL")
        mlflow.log_figure(fig, "nll.svg")

        fig, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(
            train_data, train_labels, -ll_marg_cpu, ax=ax, plot_train=False
        )
        plt.colorbar(pcm, ax=ax)
        plt.title("Marginal NLL")
        mlflow.log_figure(fig, "nll_notrain.svg")

        # logits represent P(x|y), so p(y|x) = p(x|y) * p(y) / p(x)
        posterior = logits - torch.logsumexp(logits, dim=1)[:, None]
        entropy = -torch.sum(torch.exp(posterior) * posterior, dim=1)
        # entropy = -torch.sum(torch.exp(logits) * logits, dim=1)
        entropy_cpu = entropy.cpu().detach().numpy()

        fig, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(train_data, train_labels, entropy_cpu, ax=ax)
        plt.colorbar(pcm, ax=ax)
        plt.title("Entropy")
        mlflow.log_figure(fig, "entropy.svg")

        fig, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(
            train_data, train_labels, entropy_cpu, ax=ax, plot_train=False
        )
        plt.colorbar(pcm, ax=ax)
        plt.title("Entropy")
        mlflow.log_figure(fig, "entropy_notrain.svg")

        # Einet
        test_tensor = test_tensor.to(device)
        log_posterior = model.posterior(test_tensor)
        entropy_einet = -torch.sum(torch.exp(log_posterior) * log_posterior, dim=1)
        entropy_einet_cpu = entropy_einet.cpu().detach().numpy()

        fig, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(
            train_data, train_labels, entropy_einet_cpu, ax=ax
        )
        plt.colorbar(pcm, ax=ax)
        plt.title("Entropy")
        mlflow.log_figure(fig, "entropy_einet.svg")

        fig, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(
            train_data, train_labels, entropy_einet_cpu, ax=ax, plot_train=False
        )
        plt.colorbar(pcm, ax=ax)
        plt.title("Entropy")
        mlflow.log_figure(fig, "entropy_einet_notrain.svg")

        nll_einet = -model(test_tensor).mean(dim=1)
        nll_einet_cpu = nll_einet.cpu().detach().numpy()

        fig, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(train_data, train_labels, nll_einet_cpu, ax=ax)
        plt.colorbar(pcm, ax=ax)
        plt.title("Marginal NLL")
        mlflow.log_figure(fig, "nll_einet.svg")

        fig, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(
            train_data, train_labels, nll_einet_cpu, ax=ax, plot_train=False
        )
        plt.colorbar(pcm, ax=ax)
        plt.title("Marginal NLL")
        mlflow.log_figure(fig, "nll_einet_notrain.svg")

        return 0
