import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import os
import ddu_dirty_mnist

torch.manual_seed(0)

newExp = True
# newExp = False
# load_resnet = True
load_resnet = False

batchsize = 512
data_dir = "/data_docker/datasets/"
device = "cuda" if torch.cuda.is_available() else "cpu"
result_dir = "results/dirty_mnist/"
# result_dir = "results/mnist_calib/"

mnist_transform = transforms.Compose(
    [
        # transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.reshape(-1, 28 * 28).squeeze()),
    ]
)
fashion_mnist_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.reshape(-1, 28 * 28).squeeze()),
    ]
)

# load dirty mnist
train_ds = ddu_dirty_mnist.DirtyMNIST(
    data_dir + "dirty_mnist",
    train=True,
    transform=mnist_transform,
    download=True,
    normalize=False,
    device=device,
)
valid_ds = ddu_dirty_mnist.DirtyMNIST(
    data_dir + "dirty_mnist",
    train=True,
    transform=mnist_transform,
    download=True,
    normalize=False,
    device=device,
)
test_ds = ddu_dirty_mnist.DirtyMNIST(
    data_dir + "dirty_mnist",
    train=False,
    transform=mnist_transform,
    download=True,
    normalize=False,
    device=device,
)

ambiguous_ds_test = ddu_dirty_mnist.AmbiguousMNIST(
    data_dir + "dirty_mnist",
    train=False,
    transform=mnist_transform,
    download=True,
    normalize=False,
    device=device,
)

mnist_ds_test = ddu_dirty_mnist.FastMNIST(
    data_dir + "dirty_mnist",
    train=False,
    transform=mnist_transform,
    download=True,
    normalize=False,
    device=device,
)

ood_ds = datasets.FashionMNIST(
    data_dir + "fashionmnist",
    train=False,
    download=True,
    transform=fashion_mnist_transform,
)

mean = 0.1307
std = 0.3081

train_ds, _ = torch.utils.data.random_split(
    train_ds, [0.8, 0.2], generator=torch.Generator().manual_seed(0)
)
_, valid_ds = torch.utils.data.random_split(
    valid_ds, [0.8, 0.2], generator=torch.Generator().manual_seed(0)
)
# create dataloaders
train_dl = DataLoader(
    train_ds, batch_size=batchsize, shuffle=True, pin_memory=False, num_workers=0
)
valid_dl = DataLoader(
    valid_ds, batch_size=batchsize, shuffle=True, pin_memory=False, num_workers=0
)
test_dl = DataLoader(
    test_ds, batch_size=batchsize, shuffle=True, pin_memory=False, num_workers=0
)

# create model
from ResNetSPN import ConvResNetSPN, ResidualBlockSN

from ResNetSPN import ConvResnetDDU
from net.resnet import BasicBlock

# resnet_spn = ConvResNetSPN(
#     ResidualBlockSN,
#     [2, 2, 2, 2],
#     num_classes=10,
#     image_shape=(1, 28, 28),
#     explaining_vars=[],  # for calibration test, we don't need explaining vars
#     # spec_norm_bound=6,
#     spec_norm_bound=0.9,
#     seperate_training=True,
# )
resnet_spn = ConvResnetDDU(
    BasicBlock,
    [2, 2, 2, 2],
    num_classes=10,
    image_shape=(1, 28, 28),
    spectral_normalization=True,
    mod=True,
    explaining_vars=[],  # for calibration test, we don't need explaining vars
    seperate_training=True,
)
resnet_spn = resnet_spn.to(device)

exists = os.path.isfile("resnet_spn.pt")
if not exists or newExp:
    if load_resnet:
        resnet_spn.load(result_dir + "resnet_spn.pt", resnet_only=True)
        print("loaded resnet")
    print("training resnet_spn")
    # train model
    optimizer = torch.optim.Adam(resnet_spn.parameters(), lr=0.05)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    resnet_spn.start_train(
        train_dl,
        valid_dl,
        device,
        optimizer,
        lambda_v=0.5,
        warmup_epochs=50,
        num_epochs=50,
        # warmup_epochs=50,
        # num_epochs=50,
        deactivate_resnet=True,
        lr_schedule_einet=lr_scheduler,
        early_stop=10,
        checkpoint_dir=result_dir,
    )
    resnet_spn.save(result_dir + "resnet_spn.pt")
else:
    resnet_spn.load(result_dir + "resnet_spn.pt")
    print("loaded resnet_spn.pt")

# evaluate
# resnet_spn.einet_active = False
# print("resnet accuracy train: ", resnet_spn.eval_acc(train_dl, device))
# print("resnet accuracy test: ", resnet_spn.eval_acc(test_dl, device))

resnet_spn.einet_active = True
# print("accuracy train: ", resnet_spn.eval_acc(train_dl, device))
# print("accuracy test: ", resnet_spn.eval_acc(test_dl, device))
# print("ll train: ", resnet_spn.eval_ll(train_dl, device))

# print("ll test: ", resnet_spn.eval_ll(test_dl, device))

# ood_ds_flat = ood_ds.data.reshape(-1, 28 * 28).to(dtype=torch.float32)
# ood_ds_flat = TensorDataset(ood_ds_flat, ood_ds.targets)
# ood_dl = DataLoader(ood_ds_flat, batch_size=batchsize, shuffle=True)
# print("accuracy ood: ", resnet_spn.eval_acc(ood_dl, device))
# print("ll ood: ", resnet_spn.eval_ll(ood_dl, device))

# create dataloaders
mnist_dl = DataLoader(
    mnist_ds_test, batch_size=batchsize, shuffle=False, pin_memory=False, num_workers=0
)
ambiguous_dl = DataLoader(
    ambiguous_ds_test,
    batch_size=batchsize,
    shuffle=False,
    pin_memory=False,
    num_workers=0,
)
ood_dl = DataLoader(
    ood_ds, batch_size=batchsize, shuffle=False, pin_memory=False, num_workers=0
)


# plot as in DDU paper
# three datasets: mnist, ambiguous mnist, ood
# x-axis: log-likelihood, y-axis: fraction of data

# get log-likelihoods for all datasets
with torch.no_grad():
    lls_mnist = resnet_spn.eval_ll(mnist_dl, device, return_all=True)
    pred_var_mnist = resnet_spn.eval_pred_variance(mnist_dl, device, return_all=True)
    pred_entropy_mnist = resnet_spn.eval_pred_entropy(mnist_dl, device, return_all=True)
    print("lls_mnist: ", torch.mean(lls_mnist))
    print("pred_var mnist: ", torch.mean(pred_var_mnist))
    print("pred_entropy mnist: ", torch.mean(pred_entropy_mnist))

    lls_amb = resnet_spn.eval_ll(ambiguous_dl, device, return_all=True)
    pred_var_amb = resnet_spn.eval_pred_variance(ambiguous_dl, device, return_all=True)
    pred_entropy_amb = resnet_spn.eval_pred_entropy(
        ambiguous_dl, device, return_all=True
    )
    print("lls_ambiguous: ", torch.mean(lls_amb))
    print("pred_var ambiguous: ", torch.mean(pred_var_amb))
    print("pred_entropy ambiguous: ", torch.mean(pred_entropy_amb))

    lls_ood = resnet_spn.eval_ll(ood_dl, device, return_all=True)
    pred_var_ood = resnet_spn.eval_pred_variance(ood_dl, device, return_all=True)
    pred_entropy_ood = resnet_spn.eval_pred_entropy(ood_dl, device, return_all=True)
    print("lls_ood: ", torch.mean(lls_ood))
    print("pred_var ood: ", torch.mean(pred_var_ood))
    print("pred_entropy ood: ", torch.mean(pred_entropy_ood))


# plot
def hist_plot(data_mnist, data_amb, data_ood, xlabel, ylabel, filename):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    # plot mnist
    plt.figure(figsize=(8, 6))
    sns.histplot(
        data_mnist.cpu().numpy(),
        stat="probability",
        label="MNIST",
        bins=30,
        # binrange=(min, max),
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # plot ambiguous mnist
    sns.histplot(
        data_amb.cpu().numpy(),
        stat="probability",
        label="Ambiguous MNIST",
        bins=30,
        # binrange=(min, max),
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # plot ood
    sns.histplot(
        data_ood.cpu().numpy(),
        stat="probability",
        label="Fashion MNIST",
        bins=30,
        # binrange=(min, max),
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir + filename)


# Likelihood plot
hist_plot(
    lls_mnist,
    lls_amb,
    lls_ood,
    "Log-likelihood",
    "Fraction of data",
    "mnist_amb_ood_ll.png",
)

# Predictive variance plot
hist_plot(
    pred_var_mnist,
    pred_var_amb,
    pred_var_ood,
    "Predictive variance",
    "Fraction of data",
    "mnist_amb_ood_pred_var.png",
)

# Predictive entropy plot
hist_plot(
    pred_entropy_mnist,
    pred_entropy_amb,
    pred_entropy_ood,
    "Predictive entropy",
    "Fraction of data",
    "mnist_amb_ood_pred_entropy.png",
)
