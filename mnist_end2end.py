import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from torchvision.datasets import MNIST, KMNIST, FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from ConvResNet import get_latent_batched, resnet_from_path

from simple_einet.einet import EinetConfig, Einet
from simple_einet.layers.distributions.normal import Normal

device = "cuda" if torch.cuda.is_available() else "cpu"

# newExp = True
newExp = False

# according to DDU: MNIST: id, Dirty-MNIST: id (high aleatoric), Fashion-MNIST: ood (high epistemic)
# for them: softmax entropy captures aleatoric, density-estimator captures epistemic

batchsize_resnet = 128
batchsize_einet = 128
# train_ds = MNIST(
#     "mnist", train=True, download=True, transform=ToTensor()
# )  # 60000, 28, 28
# test_ds = MNIST(
#     "mnist", train=False, download=True, transform=ToTensor()
# )  # 10000, 28, 28

train_ds_K = KMNIST("kmnist", train=True, download=True, transform=ToTensor())
test_ds_K = KMNIST("kmnist", train=False, download=True, transform=ToTensor())

train_ds = FashionMNIST("fashionmnist", train=True, download=True, transform=ToTensor())
test_ds = FashionMNIST("fashionmnist", train=False, download=True, transform=ToTensor())


def manipulate_mnist(data: np.ndarray, max_cutoff: int, noise_const: float):
    cutoffs = []
    # cutoff top rows. strong: 17, mid: 14, weak: 10
    # set row to 0
    if max_cutoff > 0:
        for i in range(data.shape[0]):
            num_cutoff = np.random.randint(max_cutoff - 5, max_cutoff)
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
    return data, np.array(cutoffs), np.array(noises)


manipulated_size = 3200
# extract some data from train_ds and test_ds
test_manipulated = test_ds.data[:manipulated_size].numpy().copy()
# test_manipulated = manipulate_mnist(test_manipulated, 0, 0.8)
test_manipulated, test_manipulated_cutoffs, test_manipulated_noises = manipulate_mnist(
    test_manipulated, 15, 0.6
)
test_manipulated = [ToTensor()(img) for img in test_manipulated]
test_manipulated_target = test_ds.targets[:manipulated_size].numpy().copy()
test_manipulated_target = [torch.tensor(target) for target in test_manipulated_target]
test_manipulated_ds = list(zip(test_manipulated, test_manipulated_target))

plt.imshow(test_manipulated_ds[0][0][0])
plt.savefig("test_manipulated.png")
plt.imshow(test_manipulated_ds[1][0][0])
plt.savefig("test_manipulated1.png")
plt.imshow(test_manipulated_ds[2][0][0])
plt.savefig("test_manipulated2.png")
print(
    "test_manipulated_target: ",
    test_manipulated_ds[0][1],
    test_manipulated_ds[1][1],
    test_manipulated_ds[2][1],
)

# also show original
plt.imshow(test_ds.data[0])
plt.savefig("test_original.png")

# show kmnist
# plt.imshow(test_ds_K.data[0])
# plt.savefig("test_original_K.png")

#### Also manipulate train data, but less intensely
train_manipulated = train_ds.data.numpy().copy()
train_manipulated, train_cutoffs, train_noises = manipulate_mnist(
    train_manipulated, 12, 0.4
)
train_manipulated = [ToTensor()(img) for img in train_manipulated]
train_manipulated_target = train_ds.targets.numpy().copy()
train_manipulated_target = [torch.tensor(target) for target in train_manipulated_target]
train_manipulated_ds = list(zip(train_manipulated, train_manipulated_target))
# print(train_manipulated.shape, cutoffs_train.shape, noises_train.shape)


train_dl = DataLoader(
    train_ds, batch_size=batchsize_resnet, shuffle=True, pin_memory=True, num_workers=1
)
test_dl = DataLoader(
    test_ds, batch_size=batchsize_resnet, pin_memory=True, num_workers=1
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
test_dl_K = DataLoader(
    test_ds_K, batch_size=batchsize_resnet, pin_memory=True, num_workers=1
)
# test_dl_F = DataLoader(test_ds_F, batch_size=batchsize_resnet, pin_memory=True, num_workers=1)

# print shape of first of test_dl
print("done loading data")

###############################################################################
from ConvResNet import ResidualBlockSN, ResNetSPN

resnet_spn = ResNetSPN(
    ResidualBlockSN, [2, 2, 2, 2], num_classes=10, spec_norm_bound=0.9
)
resnet_spn = resnet_spn.to(device)

exists = os.path.isfile("resnet_spn.pt")
if not exists or newExp:
    print("training resnet_spn")
    optimizer = torch.optim.Adam(resnet_spn.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    # start training
    epochs = 5
    for epoch in range(epochs):
        loss = 0
        for i, (images, labels) in enumerate(train_dl):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = resnet_spn(images)
            loss_v = loss_fn(outputs, labels)
            loss_v.backward()
            loss += loss_v.item()
            optimizer.step()
        print("epoch: ", epoch, " loss: ", loss / len(train_dl))
    torch.save(resnet_spn.state_dict(), "resnet_spn.pt")
else:
    resnet_spn.load_state_dict(torch.load("resnet_spn.pt"))
    print("loaded resnet_spn.pt")


def eval_acc(model, dl):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dl:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
    return correct.item() / total


def eval_ll(model, dl):
    model.eval()
    ll_total = 0
    total = 0
    with torch.no_grad():
        for images, labels in dl:
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            ll = model(images)
            ll_total += ll.mean()
    return ll_total / total


def eval_pred_softmax(model, dl):
    model.eval()
    softmax_total = 0
    total = 0
    with torch.no_grad():
        for images, labels in dl:
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            pred_logit = model(images)
            pred = torch.softmax(pred_logit, dim=1)
            softmax_total += torch.max(pred, dim=1)[0].mean()
    return softmax_total / total


def eval_variance(model, dl):
    model.eval()
    pred_var_total = 0
    total = 0
    with torch.no_grad():
        for images, labels in dl:
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            pred_logit = model(images)
            pred = torch.softmax(pred_logit, dim=1)
            pred_var = torch.var(pred, dim=1).mean()
            pred_var_total += pred_var
    return pred_var_total / total


def eval_pred_variance(model, dl):
    model.eval()
    pred_var_total = 0
    total = 0
    with torch.no_grad():
        for images, labels in dl:
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            pred_logit = model(images)
            pred = torch.softmax(pred_logit, dim=1)
            pred = torch.max(pred, dim=1)[0]
            pred_var = pred * (1 - pred)
            pred_var_total += pred_var.mean()
    return pred_var_total / total


# eval accuracies
train_acc = eval_acc(resnet_spn, train_dl)
print("train accuracy: ", train_acc)
test_acc = eval_acc(resnet_spn, test_dl)
print("test accuracy: ", test_acc)
test_manipulated_acc = eval_acc(resnet_spn, test_manipulated_dl)
print("test_manipulated accuracy: ", test_manipulated_acc)

# eval LL's
train_ll = eval_ll(resnet_spn, train_dl)
print("train_ll: ", train_ll)
test_ll = eval_ll(resnet_spn, test_dl)
print("test_ll: ", test_ll)
test_manipulated_ll = eval_ll(resnet_spn, test_manipulated_dl)
print("test_manipulated_ll: ", test_manipulated_ll)
ood_ll = eval_ll(resnet_spn, test_dl_K)
print("ood_ll: ", ood_ll)
random_data = torch.randn_like(test_manipulated[0])
random_target = torch.tensor([0])
random_ds = list(zip([random_data], [random_target]))
random_dl = DataLoader(random_ds, batch_size=1, pin_memory=True, num_workers=1)
random_ll = eval_ll(resnet_spn, random_dl)
print("random_ll: ", random_ll)

# eval softmax
train_pred_softmax = eval_pred_softmax(resnet_spn, train_dl)
print("train_pred_softmax: ", train_pred_softmax)
test_pred_softmax = eval_pred_softmax(resnet_spn, test_dl)
print("test_pred_softmax: ", test_pred_softmax)
test_manipulated_pred_softmax = eval_pred_softmax(resnet_spn, test_manipulated_dl)
print("test_manipulated_pred_softmax: ", test_manipulated_pred_softmax)
ood_pred_softmax = eval_pred_softmax(resnet_spn, test_dl_K)
print("ood_pred_softmax: ", ood_pred_softmax)
random_pred_softmax = eval_pred_softmax(resnet_spn, random_dl)
print("random_pred_softmax: ", random_pred_softmax)

# eval variance
train_variance = eval_variance(resnet_spn, train_dl)
print("train_variance: ", train_variance)
test_variance = eval_variance(resnet_spn, test_dl)
print("test_variance: ", test_variance)
test_manipulated_variance = eval_variance(resnet_spn, test_manipulated_dl)
print(test_manipulated_variance)
ood_variance = eval_variance(resnet_spn, test_dl_K)
print("ood_variance: ", ood_variance)
random_variance = eval_variance(resnet_spn, random_dl)
print("random_variance: ", random_variance)

# eval pred_variance
train_pred_variance = eval_pred_variance(resnet_spn, train_dl)
print("train_pred_variance: ", train_pred_variance)
test_pred_variance = eval_pred_variance(resnet_spn, test_dl)
print("test_pred_variance: ", test_pred_variance)
test_manipulated_pred_variance = eval_pred_variance(resnet_spn, test_manipulated_dl)
print("test_manipulated_pred_variance: ", test_manipulated_pred_variance)
ood_pred_variance = eval_pred_variance(resnet_spn, test_dl_K)
print("ood_pred_variance: ", ood_pred_variance)
random_pred_variance = eval_pred_variance(resnet_spn, random_dl)
print("random_pred_variance: ", random_pred_variance)
