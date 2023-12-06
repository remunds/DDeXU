import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch

# newExp = True
newExp = False

batch_size = 128

cifar10_url = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"
cifar10_path = "CIFAR-10-C.tar"
# download cifar10-c
if not os.path.exists(cifar10_path):
    print("Downloading CIFAR-10-C...")
    os.system(f"wget {cifar10_url} -O {cifar10_path}")

    print("Extracting CIFAR-10-C...")
    os.system(f"tar -xvf {cifar10_path}")

    print("Done!")

# get normal cifar-10
from torchvision.datasets import CIFAR10

train_ds = CIFAR10(root="cifar10", download=True, train=True)
test_ds = CIFAR10(root="cifar10", download=True, train=False)
# train: 50k, 32, 32, 3
# test: 10k, 32, 32, 3
# test-corrupted: 10k, 32, 32, 3 per corruption level (5)

corruptions = [
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    # "fog", # was broken -> reload?
    # "frost",
    # "gaussian_blur",
    # "gaussian_noise",
    # "glass_blur",
    # "impulse_noise",
    # "jpeg_compression",
    # "motion_blur",
    # "pixelate",
    # "saturate",
    # "shot_noise",
    # "snow",
    # "spatter",
    # "speckle_noise",
    # "zoom_blur",
]
corruption_ds = {}
corruption_train_ds = np.zeros((len(corruptions) * 25000, 32, 32, 3))
corruption_train_expl = np.zeros((len(corruptions) * 25000, len(corruptions)))
corruption_test_ds = np.zeros((len(corruptions) * 25000, 32, 32, 3))
corruption_test_expl = np.zeros((len(corruptions) * 25000, len(corruptions)))
corr_labels_template = np.zeros(len(corruptions))
# TODO: labels_corruptions is wrong -> compute new
for c in corruptions:
    data = np.load(f"CIFAR-10-C/{c}.npy")
    corr_labels = corr_labels_template.copy()
    corruption_index = corruptions.index(c)
    corr_labels[corruption_index] = 1
    train_i = corruption_index * 25000
    test_i = corruption_index * 25000
    for i in range(0, data.shape[0], 5000):
        curr_data = data[i : i + 5000]
        labels = ((i // 10000) + 1) * corr_labels
        is_train = i % 10000 == 0
        if is_train:
            corruption_train_ds[train_i : train_i + 5000] = curr_data
            corruption_train_expl[train_i : train_i + 5000] = labels
            train_i += 5000
        else:
            corruption_test_ds[test_i : test_i + 5000] = curr_data
            corruption_test_expl[test_i : test_i + 5000] = labels
            test_i += 5000
print("done loading corruptions")

# now add one zero for each corruption level
train_data = [ToTensor()(img).flatten() for img in train_ds.data]
train_data = torch.concat(
    [
        torch.zeros((train_ds.data.shape[0], len(corruptions))),
        torch.stack(train_data, dim=0),
    ],
    dim=1,
)
train_data = list(zip(train_data, train_ds.targets))

test_data = [ToTensor()(img).flatten() for img in test_ds.data]
test_data = torch.concat(
    [
        torch.zeros((test_ds.data.shape[0], len(corruptions))),
        torch.stack(test_data, dim=0),
    ],
    dim=1,
)
test_data = list(zip(test_data, test_ds.targets))

train_dl = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1
)
test_dl = DataLoader(
    test_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1
)

# train_corrupt: train_set + first half of each corruption
# test_corrupt: second half of each corruption
train_corrupt_data = [ToTensor()(img).flatten() for img in corruption_train_ds]
train_corrupt_data = torch.concat(
    [
        torch.from_numpy(corruption_train_expl).to(dtype=torch.int),
        torch.stack(train_corrupt_data, dim=0).to(dtype=torch.float32),
    ],
    dim=1,
)
train_corrupt_data = list(zip(train_corrupt_data, train_ds.targets))
# train_corrupt_data = train_data + train_corrupt_data
train_corrupt_dl = DataLoader(
    train_corrupt_data,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=1,
)

test_corrupt_data = [ToTensor()(img).flatten() for img in corruption_test_ds]
test_corrupt_data = torch.concat(
    [
        torch.from_numpy(corruption_test_expl).to(dtype=torch.int),
        torch.stack(test_corrupt_data, dim=0).to(dtype=torch.float32),
    ],
    dim=1,
)
test_corrupt_data = list(zip(test_corrupt_data, test_ds.targets))
# test_corrupt_data = test_data + test_corrupt_data
test_corrupt_dl = DataLoader(
    test_corrupt_data,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=1,
)


print("done loading data")
###############################################################################
from ResNetSPN import ConvResNetSPN, ResidualBlockSN

device = "cuda" if torch.cuda.is_available() else "cpu"

resnet_spn = ConvResNetSPN(
    ResidualBlockSN,
    [2, 2, 2, 2],
    num_classes=10,
    image_shape=(3, 32, 32),
    explaining_vars=list(range(len(corruptions))),
    spec_norm_bound=6,
)
resnet_spn = resnet_spn.to(device)

exists = os.path.isfile("resnet_spn_cifar.pt")
if newExp or not exists:
    print("training resnet_spn")
    optimizer = torch.optim.Adam(resnet_spn.parameters(), lr=0.01)
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.5)
    loss_fn = torch.nn.CrossEntropyLoss()
    # probably requires ~35 epochs
    # with 25 epochs, we get 0.77 train acc, 0.68 test acc
    resnet_spn.start_train(
        train_dl, device, optimizer, 1, num_epochs=25, lr_schedule=lr_schedule
    )
    resnet_spn.save("resnet_spn_cifar.pt")
else:
    resnet_spn.load("resnet_spn_cifar.pt")
    print("loaded resnet_spn.pt")

# eval accuracies
print("train acc: ", resnet_spn.eval_acc(train_dl, device))
print("test acc: ", resnet_spn.eval_acc(test_dl, device))
print("train corrupt acc: ", resnet_spn.eval_acc(train_corrupt_dl, device))
print("test corrupt acc: ", resnet_spn.eval_acc(test_corrupt_dl, device))

# eval ll's
print("train ll: ", resnet_spn.eval_ll(train_dl, device))
print("test ll: ", resnet_spn.eval_ll(test_dl, device))
print("train corrupt ll: ", resnet_spn.eval_ll(train_corrupt_dl, device))
print("test corrupt ll: ", resnet_spn.eval_ll(test_corrupt_dl, device))
