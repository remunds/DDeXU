import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torch.optim import Adam
import os

torch.manual_seed(0)

newExp = True
# newExp = False

batchsize = 512
data_dir = "/data_docker/datasets/"
device = "cuda" if torch.cuda.is_available() else "cpu"


# load mnist
train_ds = datasets.MNIST(
    data_dir + "mnist", train=True, transform=transforms.ToTensor(), download=True
)
test_ds = datasets.MNIST(
    data_dir + "mnist",
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)

ood_ds = datasets.FashionMNIST(
    data_dir + "fashionmnist",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
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

print("manipulating images randomly")
# rotate images randomly up to 45 degrees
rotations = torch.randint(0, 45, (len(train_ds),))
cutoffs = torch.randint(0, 10, (len(train_ds),))
noises = torch.randint(20, 50, (len(train_ds),))
# we want the images to be flattened (+3 for rotation, cutoff, noise)
train_ds_manip = torch.zeros((len(train_ds), 28 * 28 + 3))
for i, img in enumerate(train_ds.data):
    image = img.reshape(1, 28, 28).clone()
    # rotate image
    image = transforms.functional.rotate(
        img=image, angle=int(rotations[i]), fill=-mean / std
    )
    # cutoff top rows
    image[:, : int(cutoffs[i]), :] = 0  # -mean / std
    # add noise
    this_noise = torch.randn((1, 28, 28)) * noises[i]
    image = torch.clamp_max(image + this_noise, 255)

    train_ds_manip[i, 3:] = image.flatten()
    train_ds_manip[i, 0] = rotations[i]
    train_ds_manip[i, 1] = cutoffs[i]
    train_ds_manip[i, 2] = noises[i]

# for test_rot we want stronger rotations
# rotations = torch.randint(45, 120, (len(test_ds),))
rotations = torch.ones((len(test_ds),)) * 90
# cutoffs = torch.randint(0, 12, (len(test_ds),))
cutoffs = torch.ones((len(test_ds),)) * 1
# noises = torch.randint(20, 50, (len(test_ds),))
noises = torch.ones((len(test_ds),)) * 25

test_ds_manip = torch.zeros((len(test_ds), 28 * 28 + 3))
for i, img in enumerate(test_ds.data):
    image = img.reshape(1, 28, 28).clone()
    image = transforms.functional.rotate(
        img=image, angle=int(rotations[i]), fill=-mean / std
    )
    image[:, : int(cutoffs[i]), :] = 0
    this_noise = torch.randn((1, 28, 28)) * noises[i]
    image = torch.clamp_max(image + this_noise, 255)

    test_ds_manip[i, 3:] = image.flatten()
    test_ds_manip[i, 0] = rotations[i]
    test_ds_manip[i, 1] = cutoffs[i]
    test_ds_manip[i, 2] = noises[i]
print("done rotating images")

# show first 5 images
import matplotlib.pyplot as plt

for i in range(5):
    image = train_ds_manip[i, 3:].reshape(28, 28)
    plt.imshow(image, cmap="gray")
    plt.savefig(f"rotated_mnist_{i}.png")


# create dataloaders
train_ds_manip = TensorDataset(train_ds_manip, train_ds.targets)
train_dl = DataLoader(train_ds_manip, batch_size=batchsize, shuffle=True)
test_ds_flat = test_ds.data.reshape(-1, 28 * 28).to(dtype=torch.float32)
test_ds_flat = torch.concat((torch.zeros((len(test_ds_flat), 3)), test_ds_flat), dim=1)
test_ds_flat = TensorDataset(test_ds_flat, test_ds.targets)
test_dl = DataLoader(test_ds_flat, batch_size=batchsize, shuffle=True)
test_ds_manip = TensorDataset(test_ds_manip, test_ds.targets)
test_dl_manip = DataLoader(test_ds_manip, batch_size=batchsize, shuffle=True)

# create model
from Models import ConvResNetSPN, ResidualBlockSN

resnet_spn = ConvResNetSPN(
    ResidualBlockSN,
    [2, 2, 2, 2],
    num_classes=10,
    image_shape=(1, 28, 28),
    explaining_vars=[0, 1, 2],
    # spec_norm_bound=6,
    spec_norm_bound=0.9,
    seperate_training=True,
)
resnet_spn = resnet_spn.to(device)

exists = os.path.isfile("resnet_spn.pt")
if not exists or newExp:
    print("training resnet_spn")
    # train model
    optimizer = Adam(resnet_spn.parameters(), lr=0.01)
    resnet_spn.start_train(
        train_dl,
        device,
        optimizer,
        lambda_v=0.01,
        num_epochs=20,
        activate_einet_after=10,
        deactivate_resnet=True,
    )
    resnet_spn.save("resnet_spn.pt")
else:
    resnet_spn.load("resnet_spn.pt")
    print("loaded resnet_spn.pt")

# evaluate
resnet_spn.einet_active = False
print("resnet accuracy train: ", resnet_spn.eval_acc(train_dl, device))
print("resnet accuracy test (unrotated): ", resnet_spn.eval_acc(test_dl, device))
print("resnet accuracy test (rotated): ", resnet_spn.eval_acc(test_dl_manip, device))

resnet_spn.einet_active = True
print("accuracy train: ", resnet_spn.eval_acc(train_dl, device))
print("accuracy test (unrotated): ", resnet_spn.eval_acc(test_dl, device))
print("accuracy test (rotated): ", resnet_spn.eval_acc(test_dl_manip, device))

print("likelihood train: ", resnet_spn.eval_ll(train_dl, device))
print("likelihood test (unrotated): ", resnet_spn.eval_ll(test_dl, device))
print("likelihood test (rotated): ", resnet_spn.eval_ll(test_dl_manip, device))

ood_ds_flat = ood_ds.data.reshape(-1, 28 * 28).to(dtype=torch.float32)
ood_ds_flat = torch.concat((torch.zeros((len(ood_ds_flat), 3)), ood_ds_flat), dim=1)
ood_ds_flat = TensorDataset(ood_ds_flat, ood_ds.targets)
ood_dl = DataLoader(ood_ds_flat, batch_size=batchsize, shuffle=True)
print("accuracy ood: ", resnet_spn.eval_acc(ood_dl, device))
print("likelihood ood: ", resnet_spn.eval_ll(ood_dl, device))

print("explain LL rot: ", resnet_spn.explain_ll(test_dl_manip, device))
print("explain LL ood: ", resnet_spn.explain_ll(ood_dl, device))

print("explain MPE rot: ", resnet_spn.explain_mpe(test_dl_manip, device))
print("explain MPE ood: ", resnet_spn.explain_mpe(ood_dl, device))
