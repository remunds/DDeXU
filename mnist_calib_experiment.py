import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torch.optim import Adam
import os

torch.manual_seed(0)

# newExp = True
newExp = False
load_resnet = True
# load_resnet = False

batchsize = 512
data_dir = "/data_docker/datasets/"
device = "cuda" if torch.cuda.is_available() else "cpu"
result_dir = "results/mnist_calib/"

mnist_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.reshape(-1, 28 * 28).squeeze()),
    ]
)

# load mnist
train_ds = datasets.MNIST(
    data_dir + "mnist",
    train=True,
    transform=mnist_transform,
    download=True,
)
valid_ds = datasets.MNIST(
    data_dir + "mnist",
    train=True,
    transform=mnist_transform,
    download=True,
)

test_ds = datasets.MNIST(
    data_dir + "mnist",
    train=False,
    transform=mnist_transform,
    download=True,
)

ood_ds = datasets.FashionMNIST(
    data_dir + "fashionmnist", train=False, download=True, transform=mnist_transform
)
# train_ds = datasets.FashionMNIST(
#     data_dir + "fashionmnist",
#     train=True,
#     download=True,
#     transform=transforms.ToTensor(),
# )
# test_ds = datasets.FashionMNIST(
#     data_dir + "fashionmnist",
#     train=False,
#     download=True,
#     transform=transforms.ToTensor(),
# )
# ood_ds = datasets.MNIST(
#     data_dir + "mnist", train=False, transform=transforms.ToTensor(), download=True
# )

mean = 0.1307
std = 0.3081

print("manipulating images")
severity_levels = 5

# if we have 5k images, 1k per severity level
index_1 = len(test_ds) // severity_levels
index_2 = index_1 * 2
index_3 = index_1 * 3
index_4 = index_1 * 4


def get_severity(i):
    return (
        1
        if i < index_1
        else 2
        if i < index_2
        else 3
        if i < index_3
        else 4
        if i < index_4
        else 5
    )


# 1: 20, 2: 40, 3: 60, 4: 80, 5: 100 degrees
rotations = torch.zeros((len(test_ds),))
for i, r in enumerate(rotations):
    severity = get_severity(i)
    rotations[i] = severity * 20

# 1: 2, 2: 4, 3: 9, 4: 16, 5: 25 pixels
cutoffs = torch.zeros((len(test_ds),))
for i, r in enumerate(cutoffs):
    severity = get_severity(i)
    cutoffs[i] = 2 if severity == 1 else severity**2

# 0: 10, 1: 20, 2: 30, 3: 40, 4: 50 noise
noises = torch.zeros((len(test_ds),))
for i, r in enumerate(noises):
    severity = get_severity(i)
    noises[i] = severity * 10

test_ds_rot = torch.zeros((len(test_ds), 28 * 28))
test_ds_cutoff = torch.zeros((len(test_ds), 28 * 28))
test_ds_noise = torch.zeros((len(test_ds), 28 * 28))
for i, img in enumerate(test_ds.data):
    image = img.reshape(28, 28, 1).clone()

    this_noise = torch.randn((28, 28, 1)) * noises[i]
    img_noise = torch.clamp(image + this_noise, 0, 255).to(dtype=torch.uint8)
    img_noise = transforms.ToTensor()(img_noise.numpy())
    test_ds_noise[i] = transforms.Normalize((0.1307,), (0.3081,))(img_noise).flatten()

    image_cutoff = image.clone()
    # cutoff rows
    image_cutoff[: int(cutoffs[i]), ...] = 0
    image_cutoff = transforms.ToTensor()(image_cutoff.numpy())
    test_ds_cutoff[i] = transforms.Normalize((0.1307,), (0.3081,))(
        image_cutoff
    ).flatten()

    image_rot = transforms.ToTensor()(image.numpy())
    image_rot = transforms.functional.rotate(
        img=image_rot, angle=int(rotations[i])  # , fill=-mean / std
    )
    test_ds_rot[i] = transforms.Normalize((0.1307,), (0.3081,))(image_rot).flatten()

print("done manipulating images")

# show first 5 images
for m in ["rot", "cutoff", "noise"]:
    for i in range(5):
        import matplotlib.pyplot as plt

        image = (
            test_ds_rot[i].reshape(28, 28)
            if m == "rot"
            else test_ds_cutoff[i].reshape(28, 28)
            if m == "cutoff"
            else test_ds_noise[i].reshape(28, 28)
        )
        plt.imshow(image, cmap="gray")
        plt.savefig(f"{result_dir}{m}_{i}.png")


train_ds, _ = torch.utils.data.random_split(
    train_ds, [0.8, 0.2], generator=torch.Generator().manual_seed(0)
)
_, valid_ds = torch.utils.data.random_split(
    valid_ds, [0.8, 0.2], generator=torch.Generator().manual_seed(0)
)
# create dataloaders
train_dl = DataLoader(
    train_ds, batch_size=batchsize, shuffle=True, pin_memory=True, num_workers=2
)
valid_dl = DataLoader(
    valid_ds, batch_size=batchsize, shuffle=True, pin_memory=True, num_workers=2
)
test_dl = DataLoader(test_ds, batch_size=batchsize, shuffle=True)

# create model
# from ResNetSPN import ConvResNetSPN, ResidualBlockSN
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
    optimizer = torch.optim.Adam(resnet_spn.parameters(), lr=0.03)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    resnet_spn.start_train(
        train_dl,
        valid_dl,
        device,
        optimizer,
        lambda_v=0.8,
        # lambda_v=0.1,
        # warmup_epochs=20,
        # num_epochs=50,
        warmup_epochs=5,
        num_epochs=5,
        deactivate_resnet=True,
        lr_schedule_einet=lr_scheduler,
        early_stop=5,
        checkpoint_dir=result_dir,
    )
    resnet_spn.save(result_dir + "resnet_spn.pt")
else:
    resnet_spn.load(result_dir + "resnet_spn.pt")
    print("loaded resnet_spn.pt")

# evaluate
resnet_spn.einet_active = False
print("resnet accuracy train: ", resnet_spn.eval_acc(train_dl, device))
print("resnet accuracy test: ", resnet_spn.eval_acc(test_dl, device))

resnet_spn.einet_active = True
print("accuracy train: ", resnet_spn.eval_acc(train_dl, device))
print("accuracy test: ", resnet_spn.eval_acc(test_dl, device))
print("ll train: ", resnet_spn.eval_ll(train_dl, device))

print("ll test: ", resnet_spn.eval_ll(test_dl, device))

ood_ds_flat = ood_ds.data.reshape(-1, 28 * 28).to(dtype=torch.float32)
ood_ds_flat = TensorDataset(ood_ds_flat, ood_ds.targets)
ood_dl = DataLoader(ood_ds_flat, batch_size=batchsize, shuffle=True)
print("accuracy ood: ", resnet_spn.eval_acc(ood_dl, device))
print("ll ood: ", resnet_spn.eval_ll(ood_dl, device))

from tqdm import tqdm

# calibration test
eval_dict = {}
severity_indices = [index_1, index_2, index_3, index_4]
for s in tqdm(severity_indices):
    test_ds_rot_s = TensorDataset(test_ds_rot[:s], test_ds.targets[:s])
    test_dl_rot = DataLoader(test_ds_rot_s, batch_size=batchsize, shuffle=True)
    severity = get_severity(s - 1)
    acc = resnet_spn.eval_acc(test_dl_rot, device)
    ll = resnet_spn.eval_ll(test_dl_rot, device)
    pred_var = resnet_spn.eval_pred_variance(test_dl_rot, device)
    pred_ent = resnet_spn.eval_pred_entropy(test_dl_rot, device)
    if "rotation" not in eval_dict:
        eval_dict["rotation"] = {}
    eval_dict["rotation"][severity] = {
        "acc": acc,
        "ll": ll,
        "var": pred_var,
        "entropy": pred_ent,
    }

    test_ds_cutoff_s = TensorDataset(test_ds_cutoff[:s], test_ds.targets[:s])
    test_dl_cutoff = DataLoader(test_ds_cutoff_s, batch_size=batchsize, shuffle=True)
    severity = get_severity(s - 1)
    acc = resnet_spn.eval_acc(test_dl_cutoff, device)
    ll = resnet_spn.eval_ll(test_dl_cutoff, device)
    pred_var = resnet_spn.eval_pred_variance(test_dl_cutoff, device)
    pred_ent = resnet_spn.eval_pred_entropy(test_dl_cutoff, device)
    if "cutoff" not in eval_dict:
        eval_dict["cutoff"] = {}
    eval_dict["cutoff"][severity] = {
        "acc": acc,
        "ll": ll,
        "var": pred_var,
        "entropy": pred_ent,
    }

    test_ds_noise_s = TensorDataset(test_ds_noise[:s], test_ds.targets[:s])
    test_dl_noise = DataLoader(test_ds_noise_s, batch_size=batchsize, shuffle=True)
    severity = get_severity(s - 1)
    acc = resnet_spn.eval_acc(test_dl_noise, device)
    print("noise acc: ", acc)
    ll = resnet_spn.eval_ll(test_dl_noise, device)
    pred_var = resnet_spn.eval_pred_variance(test_dl_noise, device)
    pred_ent = resnet_spn.eval_pred_entropy(test_dl_noise, device)
    if "noise" not in eval_dict:
        eval_dict["noise"] = {}
    eval_dict["noise"][severity] = {
        "acc": acc,
        "ll": ll,
        "var": pred_var,
        "entropy": pred_ent,
    }

import numpy as np

overall_acc = np.mean(
    [eval_dict[m][severity]["acc"] for m in eval_dict for severity in eval_dict[m]]
)
overall_ll = np.mean(
    [eval_dict[m][severity]["ll"] for m in eval_dict for severity in eval_dict[m]]
)
overall_var = np.mean(
    [eval_dict[m][severity]["var"] for m in eval_dict for severity in eval_dict[m]]
)
overall_ent = np.mean(
    [eval_dict[m][severity]["entropy"] for m in eval_dict for severity in eval_dict[m]]
)
print("overall accuracy: ", overall_acc)
print("overall ll: ", overall_ll)
print("overall var: ", overall_var)
print("overall ent: ", overall_ent)


# create plot for each corruption
# x axis: severity
# y axis: acc, ll, var, entropy
import matplotlib.pyplot as plt
import numpy as np

for m in ["rotation", "cutoff", "noise"]:
    accs = [eval_dict[m][severity]["acc"] for severity in sorted(eval_dict[m].keys())]
    lls = [eval_dict[m][severity]["ll"] for severity in sorted(eval_dict[m].keys())]
    vars = [eval_dict[m][severity]["var"] for severity in sorted(eval_dict[m].keys())]
    ents = [
        eval_dict[m][severity]["entropy"] for severity in sorted(eval_dict[m].keys())
    ]
    fig, ax = plt.subplots()

    ax.set_xlabel("severity")
    ax.set_xticks(np.array(list(range(5))) + 1)

    ax.plot(accs, label="acc", color="red")
    ax.set_ylabel("accuracy", color="red")
    ax.tick_params(axis="y", labelcolor="red")
    ax.set_ylim([0, 1])

    ax2 = ax.twinx()
    ax2.plot(lls, label="ll", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")
    # ax2.set_ylim([0, 1])

    ax3 = ax.twinx()
    ax3.plot(vars, label="pred var", color="green")
    # ax3.set_ylabel("predictive variance", color="green")
    ax3.tick_params(axis="y", labelcolor="green")

    ax4 = ax.twinx()
    ax4.plot(ents, label="pred entropy", color="orange")
    # ax4.set_ylabel("predictive entropy", color="orange")
    ax4.tick_params(axis="y", labelcolor="orange")

    fig.tight_layout()
    fig.legend()
    plt.savefig(result_dir + f"{m}.png")
