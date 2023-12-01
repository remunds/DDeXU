import torch
from EinSum import EinsumExperiment
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sklearn.datasets
import os
from EinsumNetwork import EinsumNetwork

from ConvResNet import get_latent_batched, resnet_from_path
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

result_dir = "./two_moons/" 
newExp = True
# newExp = False

batchsize_resnet = 128
batchsize_einet = 128

### data and stuff from here: https://www.tensorflow.org/tutorials/understanding/sngp

### visualization macros
plt.rcParams['figure.dpi'] = 140

DEFAULT_X_RANGE = (-3.5, 3.5)
DEFAULT_Y_RANGE = (-2.5, 2.5)
DEFAULT_CMAP = colors.ListedColormap(["#377eb8", "#ff7f00"])
DEFAULT_NORM = colors.Normalize(vmin=0, vmax=1,)
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
        interpolation='bicubic',
        aspect='auto')

    # Plot training data.
    ax.scatter(train_examples[:, 0], train_examples[:, 1],
                c=train_labels, cmap=DEFAULT_CMAP, alpha=0.5)
    ax.scatter(ood_examples[:, 0], ood_examples[:, 1], c="red", alpha=0.1)

    return pcm

def make_training_data(sample_size=500):
    """Create two moon training dataset."""
    train_examples, train_labels = sklearn.datasets.make_moons(
        n_samples=2 * sample_size, noise=0.1)

    # Adjust data position slightly.
    train_examples[train_labels == 0] += [-0.1, 0.2]
    train_examples[train_labels == 1] += [0.1, -0.2]

    return train_examples.astype(np.float32), train_labels.astype(np.int32)

def make_testing_data(x_range=DEFAULT_X_RANGE, y_range=DEFAULT_Y_RANGE, n_grid=DEFAULT_N_GRID):
    """Create a mesh grid in 2D space."""
    # testing data (mesh grid over data space)
    x = np.linspace(x_range[0], x_range[1], n_grid).astype(np.float32)
    y = np.linspace(y_range[0], y_range[1], n_grid).astype(np.float32)
    xv, yv = np.meshgrid(x, y)
    return np.stack([xv.flatten(), yv.flatten()], axis=-1)

def make_ood_data(sample_size=500, means=(2.5, -1.75), vars=(0.01, 0.01)):
    return np.random.multivariate_normal(
        means, cov=np.diag(vars), size=sample_size).astype(np.float32)

# Load the train, test and OOD datasets.
train_examples, train_labels = make_training_data(
    sample_size=500)
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
test_ds = list(zip(test_examples, np.zeros((test_examples.shape[0],))))
train_dl = DataLoader(train_ds, batch_size=batchsize_resnet, shuffle=True, pin_memory=True, num_workers=1)
test_dl = DataLoader(test_ds, batch_size=batchsize_resnet, pin_memory=True, num_workers=1)

###############################################################################
exists = not newExp and os.path.isfile(f"{result_dir}latent_train.npy") and os.path.isfile(f"{result_dir}latent_test.npy") and os.path.isfile(f"{result_dir}target_train.npy") and os.path.isfile(f"{result_dir}target_test.npy") and os.path.isfile(f"{result_dir}resnet.pt") 

if exists:
    print("loading latent dataset")
    latent_train = np.load(f"{result_dir}latent_train.npy")
    target_train = np.load(f"{result_dir}target_train.npy")
    latent_test = np.load(f"{result_dir}latent_test.npy")
    target_test = np.load(f"{result_dir}target_test.npy")
    print("Latent train dataset shape: ", latent_train.shape)
    resnet = resnet_from_path(f"{result_dir}resnet.pt")
    resnet.to(device)

if not exists:
    from ResNet import ResNetSPN
    resnet_config = dict(input_dim=2, output_dim=2, num_layers=6, num_hidden=128)
    resnet = ResNetSPN(**resnet_config)
    print(resnet)
    loss_f = torch.nn.CrossEntropyLoss()
    epochs = 10
    optimizer = torch.optim.Adam(resnet.parameters(), lr=0.001)
    resnet.to(device)
    # resnet
    for epoch in range(5):
        loss = 0.0
        for data, target in train_dl:
            optimizer.zero_grad()
            target = target.type(torch.LongTensor) 
            data, target = data.to(device), target.to(device)
            output = resnet(data)
            loss_v = loss_f(output, target)
            loss += loss_v.item()
            loss_v.backward()
            optimizer.step()
        print(f"Epoch {epoch}, loss {loss / len(train_ds)}") 
    # evaluate
    resnet.eval()
    loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in train_dl:
            target = target.type(torch.LongTensor) 
            data, target = data.to(device), target.to(device)
            output = resnet(data)
            loss += loss_f(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        loss /= len(train_dl)
        print(f"Train loss: {loss}")
        print(f"Train accuracy: {correct / len(train_dl.dataset)}")
    # collect latent_data
    latent_train = []
    with torch.no_grad():
        for data, target in train_dl:
            target = target.type(torch.LongTensor) 
            data, target = data.to(device), target.to(device)
            output = resnet.forward_latent(data)
            latent_train.append(output.detach().cpu().numpy())
        latent_train = np.concatenate(latent_train, axis=0)

    latent_train = torch.from_numpy(latent_train).to(dtype=torch.float32).to(device)
    train_labels = torch.from_numpy(train_labels).to(dtype=torch.long).to(device)
    print(latent_train.shape)
    # switch to einet    
    # resnet.replace_output_layer(device)
    einsumExp = EinsumExperiment(device, latent_train.shape[1], out_dim=2)
    for epoch in range(100):
        train_ll = EinsumNetwork.eval_loglikelihood_batched(einsumExp.einet, latent_train, train_labels)
        # test_ll = EinsumNetwork.eval_loglikelihood_batched(einsumExp.einet, test_examples, np.zeros((test_examples.shape[0],)))
        print(f"Epoch {epoch}, train_ll {train_ll / len(train_ds)}")
        idx_batches = torch.randperm(latent_train.shape[0]).split(batchsize_einet)
        for batch_count, idx in enumerate(idx_batches):
            batch_x = latent_train[idx, :]
            batch_y = train_labels[idx]
            outputs = einsumExp.einet(batch_x)
            ll_sample = EinsumNetwork.log_likelihoods(outputs, batch_y)
            log_likelihood = ll_sample.sum()
            objective = log_likelihood
            objective.backward()
            einsumExp.einet.em_process_batch()
        einsumExp.einet.em_update()
    
    #evaluate
    einsumExp.eval(latent_train, train_labels, "Train")

    with torch.no_grad():
        resnet.eval()
        resnet_logits = resnet(torch.from_numpy(test_examples).to(device))
        resnet_probs = torch.nn.functional.softmax(resnet_logits, dim=1)[:, 0]
        resnet_probs = resnet_probs.cpu().numpy()
        _, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(resnet_probs, ax=ax)

        plt.colorbar(pcm, ax=ax)
        plt.title("Class Probability, SPN model")
        plt.savefig(f"{result_dir}two_moons_SPN.png")

        resnet_uncertainty = resnet_probs * (1 - resnet_probs) # predictive variance
        _, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(resnet_uncertainty, ax=ax)
        plt.colorbar(pcm, ax=ax)
        plt.title("Predictive Uncertainty, SPN Model")
        plt.savefig(f"{result_dir}two_moons_SPN_uncertainty.png")

exit()
###############################################################################

latent_test_manipulated, target_manipulated = get_latent_batched(test_manipulated_dl, manipulated_size, resnet, device, batchsize_resnet, save_dir=".")
# latent_test_K, target_test_K = get_latent_batched(test_dl_K, test_ds_K.data.shape[0], resnet, device, batchsize_resnet, save_dir=".")
# latent_test_F, target_test_F = get_latent_batched(test_dl_F, test_ds_F.data.shape[0], resnet, device, batchsize_resnet, save_dir=".")

# train_small_mlp(latent_train, target_train, latent_test, target_test, device, batchsize_resnet)


# normalize and zero-center latent space -> leads to higher accuracy and stronger LL's, but also works without
latent_train /= latent_train.max()
latent_test /= latent_test.max()
latent_test_manipulated /= latent_test_manipulated.max()
latent_train -= .5
latent_test -= .5
latent_test_manipulated -= .5

# latent_test_K /= latent_test_K.max()
# latent_test_F /= latent_test_F.max()
# latent_test_K -= .5
# latent_test_F -= .5

latent_train = torch.from_numpy(latent_train).to(dtype=torch.float32) #(N, 512)
target_train = torch.from_numpy(target_train).to(dtype=torch.long)
latent_test = torch.from_numpy(latent_test).to(dtype=torch.float32)
target_test = torch.from_numpy(target_test).to(dtype=torch.long)

# add explain-variables cutoffs and noises to latent space
# train_cutoffs = np.zeros((latent_train.shape[0], 1))
test_cutoffs = np.zeros((latent_test.shape[0], 1))
# train_noises = np.zeros((latent_train.shape[0], 1))
test_noises = np.zeros((latent_test.shape[0], 1))
latent_train = torch.cat((latent_train, torch.from_numpy(train_cutoffs).to(dtype=torch.float32).unsqueeze(1), torch.from_numpy(train_noises).to(dtype=torch.float32).unsqueeze(1)), dim=1)
latent_test = torch.cat((latent_test, torch.from_numpy(test_cutoffs).to(dtype=torch.float32), torch.from_numpy(test_noises).to(dtype=torch.float32)), dim=1)


einsumExp = EinsumExperiment(device, latent_train.shape[1])
exists = not newExp and os.path.isfile("einet.mdl") and os.path.isfile("einet.pc")
if exists:
    einsumExp.load("./")
else:
    einsumExp.train_eval(latent_train, target_train, latent_test, target_test)

einsumExp.eval(latent_train, target_train, "Train Manipulated")
einsumExp.eval(latent_test, target_test, "Test")

latent_test_manipulated = torch.from_numpy(latent_test_manipulated).to(dtype=torch.float32)

# cutoffs shape = (N,), noises shape = (N,), latent_test_manipulated shape = (N, 512)
# concat on dim=1 s.t. latent_test_manipulated shape = (N, 514)
latent_test_manipulated = torch.cat((latent_test_manipulated, torch.from_numpy(test_manipulated_cutoffs).to(dtype=torch.float32).unsqueeze(1), torch.from_numpy(test_manipulated_noises).to(dtype=torch.float32).unsqueeze(1)), dim=1)

target_manipulated = torch.from_numpy(target_manipulated).to(dtype=torch.long)
einsumExp.eval(latent_test_manipulated, target_manipulated, "Manipulated")

exp_vars = [512, 513]
exp_vals = einsumExp.explain_mpe(latent_test_manipulated[:5], exp_vars, "Manipulated")
print("exp_vals: ", exp_vals[:5])
print("orig: ", latent_test_manipulated[:5, exp_vars])

exp_vars = [512]
full_ll, marginal_ll = einsumExp.explain_ll(latent_test_manipulated, exp_vars, "Manipulated")
print("full_ll: ", full_ll)
print("marginal_ll_cutoff: ", marginal_ll)

exp_vars = [513]
full_ll, marginal_ll = einsumExp.explain_ll(latent_test_manipulated, exp_vars, "Manipulated")
print("marginal_ll_noise: ", marginal_ll)

# latent_test_K = torch.from_numpy(latent_test_K).to(dtype=torch.float32)
# target_test_K = torch.from_numpy(target_test_K).to(dtype=torch.long)
# einsumExp.eval(latent_test_K, target_test_K, "KMNIST")

# latent_test_F = torch.from_numpy(latent_test_F).to(dtype=torch.float32)
# target_test_F = torch.from_numpy(target_test_F).to(dtype=torch.long)
# einsumExp.eval(latent_test_F, target_test_F, "FashionMNIST")