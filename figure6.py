import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


def generate_data():
    samples = 1000
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


train = generate_data()
test = generate_data()

from ResNetSPN import EfficientNetSPN

model_params_dense = dict(
    num_classes=3,
    input_dim=2,
    num_layers=5,
    num_hidden=32,
    spec_norm_bound=0.95,
    einet_depth=5,
    einet_num_sums=20,
    einet_num_leaves=20,
    einet_num_repetitions=5,
    einet_leaf_type="Normal",
    einet_dropout=0.0,
)
train_params = dict(
    num_epochs=200,
    early_stop=15,
    learning_rate_warmup=0.05,
    learning_rate=0.03,
    lambda_v=0.7,
    warmup_epochs=100,
)
batch_sizes = dict(resnet=512)

from ResNetSPN import DenseResNetSPN

model = DenseResNetSPN(**model_params_dense)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_dl = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
test_dl = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)
model.start_train(
    train_dl, train_dl, device, checkpoint_dir=None, trial=None, **train_params
)
model.deactivate_uncert_head()
print("backbone: ", model.eval_acc(test_dl, device))
model.activate_uncert_head()
print("uncert: ", model.eval_acc(test_dl, device))

test_data = make_testing_data()
test_dl = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
ll_marg = model.eval_ll_marg(None, device, test_dl, return_all=True)
ll_marg = ll_marg.cpu().detach().numpy()

test_data = np.array([x[0].cpu().detach().numpy() for x in test])
test_labels = np.array([x[1].cpu().detach().numpy() for x in test])

fig, ax = plt.subplots(figsize=(7, 5.5))
pcm = plot_uncertainty_surface(test_data, test_labels, ll_marg, ax=ax)
plt.colorbar(pcm, ax=ax)
plt.title("NLL, SPN model")
plt.savefig("figure6_nll.png")

fig, ax = plt.subplots(figsize=(7, 5.5))
pcm = plot_uncertainty_surface(test_data, test_labels, ll_marg, ax=ax, plot_train=False)
plt.colorbar(pcm, ax=ax)
plt.title("NLL, SPN model")
plt.savefig("figure6_nll_notrain.png")

entropy = model.eval_entropy(None, device, test_dl, return_all=True)
entropy = entropy.cpu().detach().numpy()

fig, ax = plt.subplots(figsize=(7, 5.5))
pcm = plot_uncertainty_surface(test_data, test_labels, entropy, ax=ax)
plt.colorbar(pcm, ax=ax)
plt.title("Entropy, SPN model")
plt.savefig("figure6_entr.png")
fig, ax = plt.subplots(figsize=(7, 5.5))
pcm = plot_uncertainty_surface(test_data, test_labels, entropy, ax=ax, plot_train=False)
plt.colorbar(pcm, ax=ax)
plt.title("Entropy, SPN model")
plt.savefig("figure6_entr_notrain.png")

class_probability = model.eval_highest_class_prob(
    None, device, test_dl, return_all=True
)
class_probability = class_probability.cpu().detach().numpy()

fig, ax = plt.subplots(figsize=(7, 5.5))
pcm = plot_uncertainty_surface(test_data, test_labels, class_probability, ax=ax)
plt.colorbar(pcm, ax=ax)
plt.title("Highest class prob, SPN model")
plt.savefig("figure6_class_prob.png")

logits = model.backbone_logits(test_dl, device, return_all=True)
probs = torch.softmax(logits, dim=1)
aleatoric = -torch.sum(probs * torch.log(probs), axis=1).cpu().detach().numpy()
print(aleatoric[:5])
fig, ax = plt.subplots(figsize=(7, 5.5))
pcm = plot_uncertainty_surface(test_data, test_labels, aleatoric, ax=ax)
plt.colorbar(pcm, ax=ax)
plt.title("Aleatoric, SPN model")
plt.savefig("figure6_aleatoric.png")
