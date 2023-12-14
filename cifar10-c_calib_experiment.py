import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

torch.manual_seed(0)

newExp = True
# newExp = False
# load_resnet = True
load_resnet = False

batch_size = 512
dataset_dir = "/data_docker/datasets/"
result_dir = "/data_docker/results/cifar10-c_calib/"

device = "cuda" if torch.cuda.is_available() else "cpu"

cifar10_c_url = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"
cifar10_c_path = "CIFAR-10-C"
cifar10_c_path_complete = dataset_dir + cifar10_c_path

# download cifar10-c
if not os.path.exists(cifar10_c_path_complete + ".tar"):
    print("Downloading CIFAR-10-C...")
    os.system(f"wget {cifar10_c_url} -O {cifar10_c_path_complete}")

    print("Extracting CIFAR-10-C...")
    os.system(f"tar -xvf {cifar10_c_path_complete}.tar")

    print("Done!")

# get normal cifar-10
from torchvision.datasets import CIFAR10
from torchvision import transforms

train_transformer = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        transforms.Lambda(lambda x: x.reshape(-1, 32 * 32 * 3).squeeze()),
    ]
)
test_transformer = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        transforms.Lambda(lambda x: x.reshape(-1, 32 * 32 * 3).squeeze()),
    ]
)

train_ds = CIFAR10(
    root=dataset_dir + "cifar10",
    download=True,
    train=True,
    transform=train_transformer,
)
valid_ds = CIFAR10(
    root=dataset_dir + "cifar10", download=True, train=True, transform=test_transformer
)
train_ds, _ = torch.utils.data.random_split(
    train_ds, [45000, 5000], generator=torch.Generator().manual_seed(0)
)
_, valid_ds = torch.utils.data.random_split(
    valid_ds, [45000, 5000], generator=torch.Generator().manual_seed(0)
)

train_dl = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
valid_dl = DataLoader(
    valid_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
)

test_ds = CIFAR10(
    root=dataset_dir + "cifar10",
    download=True,
    train=False,
    transform=test_transformer,
)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# Idea: train on normal cifar10, test on (corrupted) cifar10
# Output: for each corruption, for each severity, return average accuracy and ll (including normal cifar10-test)

##############################################################33
# Training
from ResNetSPN import ConvResNetSPN, ResidualBlockSN, BottleNeckSN

resnet_spn = ConvResNetSPN(
    ResidualBlockSN,
    # BottleNeckSN,
    [2, 2, 2, 2],
    # [3, 4, 6, 3],
    num_classes=10,
    image_shape=(3, 32, 32),
    explaining_vars=[],  # for calibration experiment, no explaining vars needed
    spec_norm_bound=6,
    # spec_norm_bound=0.9,
    seperate_training=True,
)
resnet_spn = resnet_spn.to(device)

exists = os.path.isfile(result_dir + "resnet_spn.pt")
if not exists or newExp:
    if load_resnet:
        print("loading resnet")
        resnet_spn.load(result_dir + "resnet_spn.pt", resnet_only=True)
    print("training resnet_spn")
    # train model
    optimizer = Adam(resnet_spn.parameters(), lr=0.01)
    resnet_spn.start_train(
        train_dl,
        valid_dl,
        device,
        optimizer,
        # lambda_v=0.01,
        lambda_v=1,
        warmup_epochs=40,
        num_epochs=20,
        deactivate_resnet=True,
        early_stop=5,
        checkpoint_dir=result_dir,
    )
    resnet_spn.save(result_dir + "resnet_spn.pt")
else:
    resnet_spn.load(result_dir + "resnet_spn.pt")
    print("loaded resnet_spn.pt")

# Evaluate
eval_dict = {}

# eval resnet
resnet_spn.einet_active = False
train_acc = resnet_spn.eval_acc(train_dl, device)
test_acc = resnet_spn.eval_acc(test_dl, device)
print("resnet train acc: ", train_acc)
print("resnet test acc: ", test_acc)

# eval einet
resnet_spn.einet_active = True
train_acc = resnet_spn.eval_acc(train_dl, device)
train_ll = resnet_spn.eval_ll(train_dl, device)
print("train acc: ", train_acc)
print("train ll: ", train_ll)

test_acc = resnet_spn.eval_acc(test_dl, device)
test_ll = resnet_spn.eval_ll(test_dl, device)

print("test acc: ", test_acc)
print("test ll: ", test_ll)

# train: 50k, 32, 32, 3
# test: 10k, 32, 32, 3
# test-corrupted: 10k, 32, 32, 3 per corruption level (5)

corruptions = [
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",  # was broken -> reload?
    "frost",
    "gaussian_blur",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "saturate",
    "shot_noise",
    "snow",
    "spatter",
    "speckle_noise",
    "zoom_blur",
]
# iterate over all corruptions, load dataset, evaluate
for corruption in corruptions:
    print("corruption: ", corruption)
    # load dataset
    data = np.load(f"{cifar10_c_path_complete}/{corruption}.npy")
    eval_dict[corruption] = {}
    # iterate over severity levels
    for severity in range(5):
        print("severity: ", severity)
        current_data = data[severity * 10000 : (severity + 1) * 10000]
        # transform with cifar10_transformer
        current_data = torch.stack(
            [test_transformer(img) for img in current_data], dim=0
        )
        corrupt_test_ds = list(
            zip(
                current_data,
                test_ds.targets,
            )
        )
        test_dl = DataLoader(
            corrupt_test_ds,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=1,
        )

        # evaluate
        test_acc = resnet_spn.eval_acc(test_dl, device)
        test_ll = resnet_spn.eval_ll(test_dl, device)

        print("acc: ", test_acc)
        print("ll: ", test_ll)

        eval_dict[corruption][severity] = {"acc": test_acc, "ll": test_ll}

overall_acc = np.mean(
    [
        eval_dict[corruption][severity]["acc"]
        for corruption in eval_dict
        for severity in eval_dict[corruption]
    ]
)

overall_ll = np.mean(
    [
        eval_dict[corruption][severity]["ll"]
        for corruption in eval_dict
        for severity in eval_dict[corruption]
    ]
)

print("overall acc: ", overall_acc)
print("overall ll: ", overall_ll)

# create plot for each corruption
# x axis: severity
# y axis: acc, ll

import matplotlib.pyplot as plt

for corruption in eval_dict:
    accs = [
        eval_dict[corruption][severity]["acc"] for severity in eval_dict[corruption]
    ]
    lls = [eval_dict[corruption][severity]["ll"] for severity in eval_dict[corruption]]
    fig, ax = plt.subplots()
    ax.plot(accs, label="acc", color="red")
    ax.set_xlabel("severity")
    ax.set_ylabel("accuracy", color="red")
    ax.tick_params(axis="y", labelcolor="red")

    ax2 = ax.twinx()
    ax2.plot(lls, label="ll", color="blue")
    ax2.set_ylabel("log-likelihood", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")
    fig.tight_layout()
    plt.savefig(result_dir + f"{corruption}.png")
