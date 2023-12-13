import torch
import numpy as np
import os

from torchvision.datasets import MNIST, KMNIST, FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torchvision

device = "cuda" if torch.cuda.is_available() else "cpu"

newExp = True
# newExp = False

# according to DDU: MNIST: id, Dirty-MNIST: id (high aleatoric), Fashion-MNIST: ood (high epistemic)
# for them: softmax entropy captures aleatoric, density-estimator captures epistemic

batchsize_resnet = 512
batchsize_einet = 128
# train_ds = MNIST(
#     "mnist", train=True, download=True, transform=ToTensor()
# )  # 60000, 28, 28
# test_ds = MNIST(
#     "mnist", train=False, download=True, transform=ToTensor()
# )  # 10000, 28, 28

dataset_dir = "/data_docker/datasets/"
train_ds_K = KMNIST(
    dataset_dir + "kmnist", train=True, download=True, transform=ToTensor()
)
test_ds_K = KMNIST(
    dataset_dir + "kmnist", train=False, download=True, transform=ToTensor()
)

train_ds = FashionMNIST(
    dataset_dir + "fashionmnist", train=True, download=True, transform=ToTensor()
)
test_ds = FashionMNIST(
    dataset_dir + "fashionmnist", train=False, download=True, transform=ToTensor()
)


def manipulate_mnist(
    data: np.ndarray, max_cutoff: int, noise_const: float, rotation: int
):
    rotations = []
    if rotation > 0:
        rotate_values = np.random.randint(rotation // 2, rotation, data.shape[0])
        for i, rotate_value in enumerate(rotate_values):
            print(rotate_value)
            print(type(rotate_value))
            data[i] = torchvision.transforms.functional.rotate(
                img=data[i], angle=float(rotate_value)
            )
            rotations.append(rotate_value)

    cutoffs = []
    # cutoff top rows. strong: 17, mid: 14, weak: 10
    # set row to 0
    if max_cutoff > 0:
        for i in range(data.shape[0]):
            num_cutoff = np.random.randint(max_cutoff // 2, max_cutoff)
            data[i, :num_cutoff, :] = 0
            cutoffs.append(num_cutoff)
    else:
        cutoffs = np.zeros((data.shape[0],))

    noises = []
    # add noise
    if noise_const > 0:
        for i in range(data.shape[0]):
            noise = np.random.normal(0, noise_const, data.shape[1:])
            data[i] += noise.astype(np.uint8)
            noises.append(np.sum(noise))
    else:
        noises = np.zeros((data.shape[0],))
    return data, np.array(cutoffs), np.array(noises), np.array(rotations)


manipulated_size = 3200
# extract some data from train_ds and test_ds
test_manipulated = test_ds.data[:manipulated_size]
# test_manipulated = manipulate_mnist(test_manipulated, 0, 0.8)
(
    test_manipulated,
    test_manipulated_cutoffs,
    test_manipulated_noises,
    test_manipulated_rotations,
) = manipulate_mnist(test_manipulated, 0, 0, 150)
test_manipulated_cutoffs = torch.from_numpy(test_manipulated_cutoffs).view(-1, 1)
test_manipulated_noises = torch.from_numpy(test_manipulated_noises).view(-1, 1)
test_manipulated_rotations = torch.from_numpy(test_manipulated_rotations).view(-1, 1)
# test_manipulated = [ToTensor()(img).flatten() for img in test_manipulated]
test_manipulated = [img.flatten() for img in test_manipulated]
test_manipulated = torch.concat(
    [
        test_manipulated_cutoffs,
        test_manipulated_noises,
        test_manipulated_rotations,
        torch.stack(test_manipulated),
    ],
    dim=1,
).to(dtype=torch.float32)
test_manipulated_target = test_ds.targets[:manipulated_size].numpy().copy()
test_manipulated_target = [torch.tensor(target) for target in test_manipulated_target]
test_manipulated_ds = list(zip(test_manipulated, test_manipulated_target))

# plt.imshow(test_manipulated_ds[0][0][0])
# plt.savefig("test_manipulated.png")
# plt.imshow(test_manipulated_ds[1][0][0])
# plt.savefig("test_manipulated1.png")
# plt.imshow(test_manipulated_ds[2][0][0])
# plt.savefig("test_manipulated2.png")
# print(
#     "test_manipulated_target: ",
#     test_manipulated_ds[0][1],
#     test_manipulated_ds[1][1],
#     test_manipulated_ds[2][1],
# )

# # also show original
# plt.imshow(test_ds.data[0])
# plt.savefig("test_original.png")

#### Also manipulate train data, but less intensely
train_manipulated = train_ds.data.numpy().copy()
train_manipulated, train_cutoffs, train_noises, train_rotations = manipulate_mnist(
    train_manipulated, 10, 0.2, 30
)
train_cutoffs = torch.from_numpy(train_cutoffs).view(-1, 1)
train_noises = torch.from_numpy(train_noises).view(-1, 1)
train_rotations = torch.from_numpy(train_rotations).view(-1, 1)
train_manipulated = [ToTensor()(img).flatten() for img in train_manipulated]
train_manipulated = torch.concat(
    [train_cutoffs, train_noises, train_rotations, torch.stack(train_manipulated)],
    dim=1,
).to(dtype=torch.float32)
train_manipulated_target = train_ds.targets.numpy().copy()
train_manipulated_target = [torch.tensor(target) for target in train_manipulated_target]
train_manipulated_ds = list(zip(train_manipulated, train_manipulated_target))
# print(train_manipulated.shape, cutoffs_train.shape, noises_train.shape)


train_dl = DataLoader(
    train_ds, batch_size=batchsize_resnet, shuffle=True, pin_memory=True, num_workers=1
)
test_ds_complete = [ToTensor()(img).flatten() for img in test_ds.data.numpy().copy()]
test_cutoffs = torch.zeros((len(test_ds_complete), 1))
test_noises = torch.zeros((len(test_ds_complete), 1))
test_rotations = torch.zeros((len(test_ds_complete), 1))
test_ds_complete = torch.concat(
    [
        test_cutoffs,
        test_noises,
        test_rotations,
        torch.stack(test_ds_complete),
    ],
    dim=1,
).to(dtype=torch.float32)
test_ds_target = test_ds.targets.numpy().copy()
test_ds_complete = list(zip(test_ds_complete, test_ds_target))

test_dl = DataLoader(
    test_ds_complete, batch_size=batchsize_resnet, pin_memory=True, num_workers=1
)
test_manipulated_dl = DataLoader(
    test_manipulated_ds, batch_size=batchsize_resnet, pin_memory=True, num_workers=1
)
train_manipulated_dl = DataLoader(
    train_manipulated_ds,
    batch_size=batchsize_resnet,
    shuffle=True,
    pin_memory=True,
    num_workers=1,
)
train_dl = train_manipulated_dl

test_k = [ToTensor()(img).flatten() for img in test_ds_K.data.numpy().copy()]
test_k_cutoffs = torch.zeros((len(test_k), 1))
test_k_noises = torch.zeros((len(test_k), 1))
test_k_rotations = torch.zeros((len(test_k), 1))
test_k = torch.concat(
    [
        test_k_cutoffs,
        test_k_noises,
        test_k_rotations,
        torch.stack(test_k),
    ],
    dim=1,
).to(dtype=torch.float32)
test_k_target = test_ds_K.targets.numpy().copy()
test_k = list(zip(test_k, test_k_target))

test_dl_K = DataLoader(
    test_k, batch_size=batchsize_resnet, pin_memory=True, num_workers=1
)
# test_dl_F = DataLoader(test_ds_F, batch_size=batchsize_resnet, pin_memory=True, num_workers=1)

# print shape of first of test_dl
print("done loading data")

###############################################################################
from ResNetSPN import ResidualBlockSN, ConvResNetSPN, BottleNeckSN

resnet_spn = ConvResNetSPN(
    ResidualBlockSN,
    [2, 2, 2, 2],
    num_classes=10,
    image_shape=(1, 28, 28),
    explaining_vars=[0, 1, 2],
    spec_norm_bound=0.9,
)
resnet_spn = resnet_spn.to(device)

exists = os.path.isfile("resnet_spn.pt")
if not exists or newExp:
    print("training resnet_spn")
    optimizer = torch.optim.Adam(resnet_spn.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    resnet_spn.start_train(train_dl, device, optimizer, 1, 5)
    resnet_spn.save("resnet_spn.pt")
else:
    resnet_spn.load("resnet_spn.pt")
    print("loaded resnet_spn.pt")

# eval accuracies
print("train acc: ", resnet_spn.eval_acc(train_dl, device))
print("test acc: ", resnet_spn.eval_acc(test_dl, device))
print("test_manipulated acc: ", resnet_spn.eval_acc(test_manipulated_dl, device))
print("test_K acc: ", resnet_spn.eval_acc(test_dl_K, device))

# eval ll's
print("train ll: ", resnet_spn.eval_ll(train_dl, device))
print("test ll: ", resnet_spn.eval_ll(test_dl, device))
print("test_manipulated ll: ", resnet_spn.eval_ll(test_manipulated_dl, device))
print("test_K ll: ", resnet_spn.eval_ll(test_dl_K, device))

# eval dempster shafer
print("train ds: ", resnet_spn.eval_dempster_shafer(train_dl, device))
print("test ds: ", resnet_spn.eval_dempster_shafer(test_dl, device))
print(
    "test_manipulated ds: ",
    resnet_spn.eval_dempster_shafer(test_manipulated_dl, device),
)
print("test_K ds: ", resnet_spn.eval_dempster_shafer(test_dl_K, device))

# # eval predictive variance
# print("train pred var: ", resnet_spn.eval_pred_variance(train_dl, device))
# print("test pred var: ", resnet_spn.eval_pred_variance(test_dl, device))
# print(
#     "test_manipulated pred var: ",
#     resnet_spn.eval_pred_variance(test_manipulated_dl, device),
# )
# print("test_K pred var: ", resnet_spn.eval_pred_variance(test_dl_K, device))

# # eval predictive entropy
# print("train pred entropy: ", resnet_spn.eval_pred_entropy(train_dl, device))
# print("test pred entropy: ", resnet_spn.eval_pred_entropy(test_dl, device))
# print(
#     "test_manipulated pred entropy: ",
#     resnet_spn.eval_pred_entropy(test_manipulated_dl, device),
# )
# print("test_K pred entropy: ", resnet_spn.eval_pred_entropy(test_dl_K, device))

# explain via LL
explanations = resnet_spn.explain_ll(test_manipulated_dl, device)
print("LL explanations: ", explanations)

# explain via MPE
small_test_manip = test_manipulated_dl.dataset[:10]
small_test_manip_dl = DataLoader(small_test_manip, batch_size=2)
explanations = resnet_spn.explain_mpe(small_test_manip_dl, device)
print("MPE explanations: ", explanations)
