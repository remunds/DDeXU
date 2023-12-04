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
# train_ds = MNIST("mnist", train=True, download=True, transform=ToTensor()) # 60000, 28, 28
# test_ds = MNIST("mnist", train=False, download=True, transform=ToTensor()) # 10000, 28, 28

# train_ds = KMNIST("kmnist", train=True, download=True, transform=ToTensor())
# test_ds = KMNIST("kmnist", train=False, download=True, transform=ToTensor())

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
    test_manipulated, 20, 0.0
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
# test_dl_K = DataLoader(test_ds_K, batch_size=batchsize_resnet, pin_memory=True, num_workers=1)
# test_dl_F = DataLoader(test_ds_F, batch_size=batchsize_resnet, pin_memory=True, num_workers=1)

# print shape of first of test_dl
print("done loading data")

###############################################################################
exists = (
    not newExp
    and os.path.isfile("latent_train.npy")
    and os.path.isfile("latent_test.npy")
    and os.path.isfile("target_train.npy")
    and os.path.isfile("target_test.npy")
    and os.path.isfile("./resnet.pt")
)
# exists = True

if exists:
    print("loading latent dataset")
    latent_train = np.load("latent_train.npy")
    target_train = np.load("target_train.npy")
    latent_test = np.load("latent_test.npy")
    target_test = np.load("target_test.npy")
    print("Latent train dataset shape: ", latent_train.shape)
    resnet = resnet_from_path("./resnet.pt")
    resnet.to(device)

if not exists:
    from ConvResNet import train_eval_resnet, get_latent_dataset

    resnet = train_eval_resnet(10, train_dl, test_dl, device, save_dir=".")
    latent_train, target_train, latent_test, target_test = get_latent_dataset(
        train_ds,
        train_dl,
        test_ds,
        test_dl,
        resnet,
        device,
        batchsize_resnet,
        save_dir=".",
    )

print("collecting manipulated test latent space dataset")
latent_test_manipulated, target_manipulated = get_latent_batched(
    test_manipulated_dl,
    manipulated_size,
    resnet,
    device,
    batchsize_resnet,
    save_dir=".",
)
print("done")
# latent_test_K, target_test_K = get_latent_batched(test_dl_K, test_ds_K.data.shape[0], resnet, device, batchsize_resnet, save_dir=".")
# latent_test_F, target_test_F = get_latent_batched(test_dl_F, test_ds_F.data.shape[0], resnet, device, batchsize_resnet, save_dir=".")

# train_small_mlp(latent_train, target_train, latent_test, target_test, device, batchsize_resnet)


# normalize and zero-center latent space -> leads to higher accuracy and stronger LL's, but also works without
latent_train /= latent_train.max()
latent_test /= latent_test.max()
latent_test_manipulated /= latent_test_manipulated.max()
latent_train -= 0.5
latent_test -= 0.5
latent_test_manipulated -= 0.5

# latent_test_K /= latent_test_K.max()
# latent_test_F /= latent_test_F.max()
# latent_test_K -= .5
# latent_test_F -= .5

latent_train = torch.from_numpy(latent_train).to(dtype=torch.float32)  # (N, 512)
target_train = torch.from_numpy(target_train).to(dtype=torch.long)
latent_test = torch.from_numpy(latent_test).to(dtype=torch.float32)
target_test = torch.from_numpy(target_test).to(dtype=torch.long)

# add explain-variables cutoffs and noises to latent space
# train_cutoffs = np.zeros((latent_train.shape[0], 1))
test_cutoffs = np.zeros((latent_test.shape[0], 1))
# train_noises = np.zeros((latent_train.shape[0], 1))
test_noises = np.zeros((latent_test.shape[0], 1))
latent_train = torch.cat(
    (
        latent_train,
        torch.from_numpy(train_cutoffs).to(dtype=torch.float32).unsqueeze(1),
        torch.from_numpy(train_noises).to(dtype=torch.float32).unsqueeze(1),
    ),
    dim=1,
)
latent_test = torch.cat(
    (
        latent_test,
        torch.from_numpy(test_cutoffs).to(dtype=torch.float32),
        torch.from_numpy(test_noises).to(dtype=torch.float32),
    ),
    dim=1,
)
latent_test_manipulated = torch.cat(
    (
        torch.from_numpy(latent_test_manipulated).to(dtype=torch.float32),
        torch.from_numpy(test_manipulated_cutoffs).to(dtype=torch.float32).unsqueeze(1),
        torch.from_numpy(test_manipulated_noises).to(dtype=torch.float32).unsqueeze(1),
    ),
    dim=1,
)


cfg = EinetConfig(
    num_features=latent_train.shape[1],
    num_channels=1,
    depth=3,
    num_sums=20,
    num_leaves=20,
    # num_repetitions=20,
    num_repetitions=1,
    num_classes=10,
    leaf_type=Normal,
    # leaf_kwargs={"total_count": 2**n_bits - 1},
    layer_type="einsum",
    dropout=0.0,
)
model = Einet(cfg)
model.to(device)
print(model)

print(latent_train.shape, target_train.shape, latent_test.shape, target_test.shape)
print(latent_test_manipulated.shape, target_manipulated.shape)


def accuracy(model, X, y):
    with torch.no_grad():
        correct = 0
        for i in range(0, X.shape[0], batchsize_resnet):
            data = X[i : i + batchsize_resnet].to(device)
            target = y[i : i + batchsize_resnet].to(device)
            outputs = model(data)
            predictions = outputs.argmax(-1)
            correct += (predictions == target).sum().item()
        total = y.shape[0]
        return correct / total


def log_likelihood(model, X):
    with torch.no_grad():
        ll = 0
        for i in range(0, X.shape[0], batchsize_resnet):
            data = X[i : i + batchsize_resnet].to(device)
            outputs = model(data)
            ll += outputs.mean()
        return ll / (X.shape[0] / batchsize_resnet)

exists = (not newExp and os.path.isfile("./einet.pt"))
if not exists:
    lr = 0.01
    epochs = 5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        milestones=[int(epochs * 0.7), int(epochs * 0.9)], optimizer=optimizer, gamma=0.5
    )
    cross_entropy = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        loss = 0
        total_ll = 0
        for i in range(0, latent_train.shape[0], batchsize_resnet):
            optimizer.zero_grad()
            latent_train_batch = latent_train[i : i + batchsize_resnet].to(device)
            target_train_batch = target_train[i : i + batchsize_resnet].to(device)
            lls = model(latent_train_batch)
            total_ll += lls.mean().item()
            # if epoch > 2:
            loss_v = cross_entropy(lls, target_train_batch)
            # else:
            #     loss_v = -lls.mean()
            loss_v.backward()
            optimizer.step()
            loss += loss_v.item()
        lr_scheduler.step()
        test_acc = accuracy(model, latent_test, target_test)
        print(
            "Epoch: ",
            epoch,
            "Train LL: ",
            total_ll / (latent_train.shape[0] / batchsize_resnet),
            "Train Loss: ",
            loss / (latent_train.shape[0] / batchsize_resnet),
            "Test accuracy: ",
            test_acc,
        )
    torch.save(model.state_dict(), "./einet.pt")
else:
    model.load_state_dict(torch.load("./einet.pt"))
model.eval()


print("Accuracy on train set: ", accuracy(model, latent_train, target_train))
print("Accuracy on test set: ", accuracy(model, latent_test, target_test))
target_manipulated = torch.from_numpy(target_manipulated)
print(
    "Accuracy on manipulated test set: ",
    accuracy(model, latent_test_manipulated, target_manipulated),
)
print("LL on train set: ", log_likelihood(model, latent_train))
print("LL on test set: ", log_likelihood(model, latent_test))
print(
    "LL on manipulated test set: ",
    log_likelihood(model, latent_test_manipulated),
)
ood = torch.randn_like(latent_test_manipulated)
print("LL on ood test set: ", log_likelihood(model, ood))

mpe_input = latent_test_manipulated[:5]
# mpe_input[:, -2:] = 0.0#np.nan
marginalized_scopes = [512, 513]
mpe_input = mpe_input.to(device)
print(mpe_input.shape)
sample_out = model.sample(evidence=mpe_input, marginalized_scopes=marginalized_scopes)
mpe_leave_out = model.sample(evidence=mpe_input, marginalized_scopes=marginalized_scopes, mpe_at_leaves=True)
mpe_out = model.mpe(evidence=mpe_input, marginalized_scopes=marginalized_scopes)
print("before: ", mpe_input[:5, marginalized_scopes])
print("sample_out: ", sample_out[:5, marginalized_scopes])
print("mpe_leave_out: ", mpe_leave_out[:5, marginalized_scopes])
print("mpe_out: ", mpe_out[:5, marginalized_scopes])

# cutoffs shape = (N,), noises shape = (N,), latent_test_manipulated shape = (N, 512)
# concat on dim=1 s.t. latent_test_manipulated shape = (N, 514)
# latent_test_manipulated = torch.cat(
#     (
#         latent_test_manipulated,
#         torch.from_numpy(test_manipulated_cutoffs).to(dtype=torch.float32).unsqueeze(1),
#         torch.from_numpy(test_manipulated_noises).to(dtype=torch.float32).unsqueeze(1),
#     ),
#     dim=1,
# )

# target_manipulated = torch.from_numpy(target_manipulated).to(dtype=torch.long)
# einsumExp.eval(latent_test_manipulated, target_manipulated, "Manipulated")

# exp_vars = [512, 513]
# exp_vals = einsumExp.explain_mpe(latent_test_manipulated[:5], exp_vars, "Manipulated")
# print("exp_vals: ", exp_vals[:5])
# print("orig: ", latent_test_manipulated[:5, exp_vars])

# exp_vars = [512]
# full_ll, marginal_ll = einsumExp.explain_ll(
#     latent_test_manipulated, exp_vars, "Manipulated"
# )
# print("full_ll: ", full_ll)
# print("marginal_ll_cutoff: ", marginal_ll)

# exp_vars = [513]
# full_ll, marginal_ll = einsumExp.explain_ll(
#     latent_test_manipulated, exp_vars, "Manipulated"
# )
# print("marginal_ll_noise: ", marginal_ll)

# latent_test_K = torch.from_numpy(latent_test_K).to(dtype=torch.float32)
# target_test_K = torch.from_numpy(target_test_K).to(dtype=torch.long)
# einsumExp.eval(latent_test_K, target_test_K, "KMNIST")

# latent_test_F = torch.from_numpy(latent_test_F).to(dtype=torch.float32)
# target_test_F = torch.from_numpy(target_test_F).to(dtype=torch.long)
# einsumExp.eval(latent_test_F, target_test_F, "FashionMNIST")
