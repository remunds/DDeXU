import os
import numpy as np
from torch.utils.data import DataLoader
import torch

torch.manual_seed(0)

newExp = True
# newExp = False

batch_size = 512
dataset_dir = "/data_docker/datasets/"
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

cifar10_transformer = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
)

train_ds = CIFAR10(root=dataset_dir + "cifar10", download=True, train=True)
test_ds = CIFAR10(root=dataset_dir + "cifar10", download=True, train=False)

# train: 50k, 32, 32, 3
# test: 10k, 32, 32, 3
# test-corrupted: 10k, 32, 32, 3 per corruption level (5)

corruptions = [
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    # "fog", # was broken -> reload?
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
# now add one zero for each corruption level
train_data = [cifar10_transformer(img).flatten() for img in train_ds.data]
train_data = torch.concat(
    [
        torch.zeros((train_ds.data.shape[0], len(corruptions))),
        torch.stack(train_data, dim=0),
    ],
    dim=1,
)
train_data = list(zip(train_data, train_ds.targets))

test_data = [cifar10_transformer(img).flatten() for img in test_ds.data]
test_data = torch.concat(
    [
        torch.zeros((test_ds.data.shape[0], len(corruptions))),
        torch.stack(test_data, dim=0),
    ],
    dim=1,
)
test_data = list(zip(test_data, test_ds.targets))

# train_dl = DataLoader(
#     train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1
# )
test_dl = DataLoader(
    test_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1
)


def get_corrupted_cifar10(
    all_corruption_len: int, corruptions: list, test_labels: np.ndarray, levels: list
):
    # available corrupted dataset: 50k*len(corruptions), 32, 32, 3
    datasets_length = 10000 * len(levels) * len(corruptions) // 2
    train_corrupt_data = np.zeros((datasets_length, 32, 32, 3), dtype=np.uint8)
    test_corrupt_data = np.zeros((datasets_length, 32, 32, 3), dtype=np.uint8)
    train_corrupt_levels = np.zeros(
        (datasets_length, all_corruption_len), dtype=np.uint8
    )
    test_corrupt_levels = np.zeros(
        (datasets_length, all_corruption_len), dtype=np.uint8
    )
    train_corrupt_labels = np.zeros((datasets_length), dtype=np.uint8)
    test_corrupt_labels = np.zeros((datasets_length) + 5000, dtype=np.uint8)
    train_idx = 0
    test_idx = 0
    for corr_idx, c in enumerate(corruptions):
        # each corrupted dataset has shape of test: 50k, 32, 32, 3
        data = np.load(f"{cifar10_c_path_complete}/{c}.npy")
        # step in 5000s, because each corruption level has 5000 images
        for i in range(5):  # iterate over corruption levels
            if not i in levels:
                continue
            data_idx = i * 10000
            new_train_idx = train_idx + 5000
            new_test_idx = test_idx + 5000
            train_corrupt_data[train_idx:new_train_idx] = data[
                data_idx : (data_idx + 5000), ...
            ]
            train_corrupt_levels[train_idx:new_train_idx, corr_idx] = i + 1
            train_corrupt_labels[train_idx:new_train_idx] = test_labels[:5000]

            test_corrupt_data[test_idx:new_test_idx] = data[
                (data_idx + 5000) : (data_idx + 10000), ...
            ]
            test_corrupt_levels[test_idx:new_test_idx, corr_idx] = i + 1
            test_corrupt_labels[test_idx:new_test_idx] = test_labels[5000:]
            train_idx = new_train_idx
            test_idx = new_test_idx

    print("done loading corruptions")
    return (
        train_corrupt_data,
        train_corrupt_levels,
        train_corrupt_labels,
        test_corrupt_data,
        test_corrupt_levels,
        test_corrupt_labels,
    )


print("loading corrupted data")
(
    train_corrupt_data,
    train_corrupt_levels,
    train_corrupt_labels,
    test_corrupt_data,
    test_corrupt_levels,
    test_corrupt_labels,
) = get_corrupted_cifar10(
    len(corruptions), ["gaussian_noise", "snow"], np.array(test_ds.targets), [1, 2]
)
train_corrupt_data = [cifar10_transformer(img).flatten() for img in train_corrupt_data]
train_corrupt_data = torch.concat(
    [
        torch.from_numpy(train_corrupt_levels).to(dtype=torch.int32),
        torch.stack(train_corrupt_data, dim=0).to(dtype=torch.float32),
    ],
    dim=1,
)
train_corrupt_data = list(zip(train_corrupt_data, train_corrupt_labels))
# We want to train on some of the corrupted data, s.t. explanations are possible
train_data_combined = train_data + train_corrupt_data
train_dl = DataLoader(
    train_data_combined,
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
    seperate_training=True,
)

resnet_spn = resnet_spn.to(device)

exists = os.path.isfile("resnet_spn_cifar.pt")
if newExp or not exists:
    print("training resnet_spn")
    optimizer = torch.optim.Adam(resnet_spn.parameters(), lr=0.01)
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.5)
    # probably requires ~35 epochs
    # with 25 epochs, we get 0.77 train acc, 0.68 test acc
    resnet_spn.start_train(
        train_dl,
        device,
        optimizer,
        lambda_v=0.3,
        num_epochs=30,
        activate_einet_after=10,
        deactivate_resnet_with_einet_train=True,
        lr_schedule=lr_schedule,
    )
    resnet_spn.save("resnet_spn_cifar.pt")
else:
    resnet_spn.load("resnet_spn_cifar.pt")
    print("loaded resnet_spn.pt")


# eval accuracies
print("train acc: ", resnet_spn.eval_acc(train_dl, device))
print("train ll: ", resnet_spn.eval_ll(train_dl, device))

print("test acc: ", resnet_spn.eval_acc(test_dl, device))
print("test ll: ", resnet_spn.eval_ll(test_dl, device))

del train_dl
del test_dl

train_corrupt_dl = DataLoader(
    train_corrupt_data,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=1,
)
print("train corrupt acc: ", resnet_spn.eval_acc(train_corrupt_dl, device))
print("train corrupt ll: ", resnet_spn.eval_ll(train_corrupt_dl, device))
del train_corrupt_dl

print("loading more corrupted data")
(
    train_corrupt_data,
    train_corrupt_levels,
    train_corrupt_labels,
    test_corrupt_data,
    test_corrupt_levels,
    test_corrupt_labels,
) = get_corrupted_cifar10(
    len(corruptions), ["gaussian_noise", "snow"], np.array(test_ds.targets), [5]
)
test_corrupt_data = [cifar10_transformer(img).flatten() for img in test_corrupt_data]
test_corrupt_data = torch.concat(
    [
        # torch.from_numpy(test_corrupt_levels).to(dtype=torch.int32),
        torch.zeros_like(torch.from_numpy(test_corrupt_levels)).to(dtype=torch.int32),
        torch.stack(test_corrupt_data, dim=0).to(dtype=torch.float32),
    ],
    dim=1,
)
test_corrupt_data = list(zip(test_corrupt_data, test_corrupt_labels))
test_corrupt_dl = DataLoader(
    test_corrupt_data,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=1,
)

print("test corrupt acc: ", resnet_spn.eval_acc(test_corrupt_dl, device))
print("test corrupt ll: ", resnet_spn.eval_ll(test_corrupt_dl, device))

print("explain corrupt ll: ", resnet_spn.explain_ll(test_corrupt_dl, device))
print("explain corrupt mpe: ", resnet_spn.explain_mpe(test_corrupt_dl, device))
