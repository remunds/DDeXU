import torch
from EinSum import EinsumExperiment
import numpy as np
import matplotlib.pyplot as plt
import os

from ResNetHidden import get_latent_batched, resnet_from_path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# according to DDU: MNIST: id, Dirty-MNIST: id (high aleatoric), Fashion-MNIST: ood (high epistemic)
# for them: softmax entropy captures aleatoric, density-estimator captures epistemic

from torchvision.datasets import MNIST, KMNIST, FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

batchsize_resnet = 64
train_ds = MNIST("mnist", train=True, download=True, transform=ToTensor()) # 60000, 28, 28
test_ds = MNIST("mnist", train=False, download=True, transform=ToTensor()) # 10000, 28, 28

# train_ds_K = KMNIST("kmnist", train=True, download=True, transform=ToTensor()) 
# test_ds_K = KMNIST("kmnist", train=False, download=True, transform=ToTensor()) 

# train_ds_F = FashionMNIST("fashionmnist", train=True, download=True, transform=ToTensor())
# test_ds_F = FashionMNIST("fashionmnist", train=False, download=True, transform=ToTensor())

def manipulate_mnist(data: np.ndarray, num_cutoff: int, noise_const: float):
    # cutoff top rows. strong: 17, mid: 14, weak: 10 
    # set row to 0
    if num_cutoff > 0:
        data[:, :num_cutoff, :] = 0

    # add noise
    if noise_const > 0:
        # normal noise between 0 and 255
        noise = np.random.normal(0, noise_const, data.shape)
        data += noise.astype(np.uint8)

    return data

manipulated_size = 3200
# extract some data from train_ds and test_ds
test_manipulated = test_ds.data[:manipulated_size].numpy().copy()
# test_manipulated = manipulate_mnist(test_manipulated, 0, 0.8)
test_manipulated = manipulate_mnist(test_manipulated, 12, 1)
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
print("test_manipulated_target: ", test_manipulated_ds[0][1], test_manipulated_ds[1][1], test_manipulated_ds[2][1])

# also show original
plt.imshow(test_ds.data[0])
plt.savefig("test_original.png")

# show kmnist
# plt.imshow(test_ds_K.data[0])
# plt.savefig("test_original_K.png")

train_dl = DataLoader(train_ds, batch_size=batchsize_resnet, shuffle=True, pin_memory=True, num_workers=1)
test_dl = DataLoader(test_ds, batch_size=batchsize_resnet, pin_memory=True, num_workers=1)
test_manipulated_dl = DataLoader(test_manipulated_ds, batch_size=batchsize_resnet, pin_memory=True, num_workers=1)
# test_dl_K = DataLoader(test_ds_K, batch_size=batchsize_resnet, pin_memory=True, num_workers=1)
# test_dl_F = DataLoader(test_ds_F, batch_size=batchsize_resnet, pin_memory=True, num_workers=1)

# print shape of first of test_dl
print("done loading data")

###############################################################################
exists = os.path.isfile("latent_train.npy") and os.path.isfile("latent_test.npy") and os.path.isfile("target_train.npy") and os.path.isfile("target_test.npy") and os.path.isfile("./resnet.pt")
# exists = False

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
    from ResNetHidden import train_eval_resnet, get_latent_dataset
    resnet = train_eval_resnet(train_dl, test_dl, device, save_dir=".") 
    latent_train, target_train, latent_test, target_test = get_latent_dataset(train_ds, train_dl, test_ds, test_dl, resnet, device, batchsize_resnet, save_dir=".")

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

latent_train = torch.from_numpy(latent_train).to(dtype=torch.float32)
target_train = torch.from_numpy(target_train).to(dtype=torch.long)
latent_test = torch.from_numpy(latent_test).to(dtype=torch.float32)
target_test = torch.from_numpy(target_test).to(dtype=torch.long)

einsumExp = EinsumExperiment(device, latent_train.shape[1])
exists = os.path.isfile("einet.mdl") and os.path.isfile("einet.pc")
if exists:
    einsumExp.load("./")
else:
    einsumExp.train_eval(latent_train, target_train, latent_test, target_test)

einsumExp.eval(latent_test, target_test, "Test")

latent_test_manipulated = torch.from_numpy(latent_test_manipulated).to(dtype=torch.float32)
target_manipulated = torch.from_numpy(target_manipulated).to(dtype=torch.long)
einsumExp.eval(latent_test_manipulated, target_manipulated, "Manipulated")

# latent_test_K = torch.from_numpy(latent_test_K).to(dtype=torch.float32)
# target_test_K = torch.from_numpy(target_test_K).to(dtype=torch.long)
# einsumExp.eval(latent_test_K, target_test_K, "KMNIST")

# latent_test_F = torch.from_numpy(latent_test_F).to(dtype=torch.float32)
# target_test_F = torch.from_numpy(target_test_F).to(dtype=torch.long)
# einsumExp.eval(latent_test_F, target_test_F, "FashionMNIST")