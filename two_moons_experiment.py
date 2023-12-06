import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sklearn.datasets

from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

result_dir = "./two_moons/"
# newExp = True
newExp = False

batchsize_resnet = 128
batchsize_einet = 128

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


def plot_uncertainty_surface(test_uncertainty, ax, cmap=None):
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
    if np.max(test_uncertainty) > 0:
        test_uncertainty = test_uncertainty / np.max(test_uncertainty)
    else:
        test_uncertainty = test_uncertainty / np.min(test_uncertainty)

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


# Load the train, test and OOD datasets.
train_examples, train_labels = make_training_data(sample_size=500)
test_examples = make_testing_data()
ood_examples = make_ood_data(sample_size=500)

# Visualize
pos_examples = train_examples[train_labels == 0]
neg_examples = train_examples[train_labels == 1]

plt.figure(figsize=(7, 5.5))

plt.scatter(pos_examples[:, 0], pos_examples[:, 1], c="#377eb8", alpha=0.5)
plt.scatter(neg_examples[:, 0], neg_examples[:, 1], c="#ff7f00", alpha=0.5)
plt.scatter(ood_examples[:, 0], ood_examples[:, 1], c="red", alpha=0.1)

plt.legend(["Positive", "Negative", "Out-of-Domain"])

plt.ylim(DEFAULT_Y_RANGE)
plt.xlim(DEFAULT_X_RANGE)

plt.savefig(f"{result_dir}two_moons.png")
# put into data loaders
train_ds = list(zip(train_examples, train_labels))
train_dl = DataLoader(
    train_ds, batch_size=batchsize_resnet, shuffle=True, pin_memory=True, num_workers=1
)

###############################################################################
from ResNetSPN import DenseResNetSPN

resnet_config = dict(input_dim=2, output_dim=2, num_layers=3, num_hidden=32)
resnet = DenseResNetSPN(**resnet_config)
print(resnet)
epochs = 50
optimizer = torch.optim.Adam(resnet.parameters(), lr=0.03)
resnet.to(device)
# resnet
resnet.start_train(train_dl, device, optimizer, 1, epochs)
# evaluate
print("accuracy: ", resnet.eval_acc(train_dl, device))

with torch.no_grad():
    resnet.eval()

    # get LL's of first 5 training examples
    pos_ll = resnet(torch.from_numpy(pos_examples[:5]).to(device))
    print("pos_ll: ", pos_ll)
    neg_ll = resnet(torch.from_numpy(neg_examples[:5]).to(device))
    print("neg_ll: ", neg_ll)

    # get LL's 5 ood examples
    ood_ll = resnet(torch.from_numpy(ood_examples[:5]).to(device))
    print("ood_ll: ", ood_ll)

    resnet_logits = resnet(torch.from_numpy(test_examples).to(device))
    resnet_probs = torch.nn.functional.softmax(resnet_logits, dim=1)[:, 0]
    resnet_probs = resnet_probs.cpu().numpy()
    _, ax = plt.subplots(figsize=(7, 5.5))
    pcm = plot_uncertainty_surface(resnet_probs, ax=ax)

    plt.colorbar(pcm, ax=ax)
    plt.title("Class Probability, SPN model")
    plt.savefig(f"{result_dir}two_moons_SPN.png")

    resnet_uncertainty = resnet_probs * (1 - resnet_probs)  # predictive variance
    _, ax = plt.subplots(figsize=(7, 5.5))
    pcm = plot_uncertainty_surface(resnet_uncertainty, ax=ax)
    plt.colorbar(pcm, ax=ax)
    plt.title("Predictive Uncertainty, SPN Model")
    plt.savefig(f"{result_dir}two_moons_SPN_uncertainty.png")

    resnet_uncertainty = resnet_logits.cpu().numpy()[:, 0]  # log likelihood
    print(resnet_uncertainty[:5])
    _, ax = plt.subplots(figsize=(7, 5.5))
    pcm = plot_uncertainty_surface(resnet_uncertainty, ax=ax)
    plt.colorbar(pcm, ax=ax)
    plt.title("LL Uncertainty, SPN Model")
    plt.savefig(f"{result_dir}two_moons_SPN_ll_uncertainty.png")

    print(test_examples.shape)
    test_dl = DataLoader(
        test_examples, batch_size=batchsize_resnet, pin_memory=True, num_workers=1
    )
    resnet_uncertainty = resnet.eval_dempster_shafer(test_dl, device)
    print(resnet_uncertainty.shape)
    print(resnet_uncertainty[:5])
    _, ax = plt.subplots(figsize=(7, 5.5))
    pcm = plot_uncertainty_surface(resnet_uncertainty, ax=ax)
    plt.colorbar(pcm, ax=ax)
    plt.title("Dempster Shafer, SPN Model")
    plt.savefig(f"{result_dir}two_moons_SPN_dempster_shafer.png")
