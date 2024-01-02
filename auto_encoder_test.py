import torch
from torchvision import datasets, transforms

batchsize = 512
data_dir = "/data_docker/datasets/"
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

# load data
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

train_ds, _ = torch.utils.data.random_split(
    train_ds, [0.8, 0.2], generator=torch.Generator().manual_seed(0)
)
_, valid_ds = torch.utils.data.random_split(
    valid_ds, [0.8, 0.2], generator=torch.Generator().manual_seed(0)
)


train_dl = torch.utils.data.DataLoader(
    train_ds, batch_size=batchsize, shuffle=True, pin_memory=True, num_workers=2
)
valid_dl = torch.utils.data.DataLoader(
    valid_ds, batch_size=batchsize, shuffle=True, pin_memory=True, num_workers=1
)
test_dl = torch.utils.data.DataLoader(
    test_ds, batch_size=batchsize, shuffle=True, pin_memory=True, num_workers=1
)
ood_dl = torch.utils.data.DataLoader(
    ood_ds, batch_size=batchsize, shuffle=True, pin_memory=True, num_workers=1
)


# train autoencoder
from ResNetSPN import AutoEncoderSPN

# import CrossEntropyLoss
from tqdm import tqdm

autoenc = AutoEncoderSPN()
autoenc.to(device)

# optimizer = torch.optim.Adam(autoenc.parameters(), lr=0.0005)
# for epoch in range(30):
#     for x, y in tqdm(train_dl):
#         optimizer.zero_grad()
#         x, y = x.to(device), y.to(device)
#         lls = autoenc(x)
#         ce_loss = torch.nn.CrossEntropyLoss()(lls, y)
#         # get 64 random numbers between 0 and 128, no duplicates
#         marginalized_scope = torch.randperm(128)[:32].tolist()
#         sample = autoenc.einet.sample(
#             evidence=autoenc.latent,
#             marginalized_scopes=marginalized_scope,
#             is_differentiable=True,
#             is_mpe=True,
#         )
#         # mpe = autoenc.einet.sample(x.shape[0], is_mpe=True, is_differentiable=True)
#         # reconstruct image
#         recon = autoenc.decode(sample)
#         # compute loss
#         loss_recon = torch.nn.MSELoss()(recon, x)
#         loss = ce_loss + loss_recon
#         loss.backward()
#         optimizer.step()

#     # print accuracy
#     autoenc.eval()
#     with torch.no_grad():
#         acc = 0
#         for x, y in tqdm(valid_dl):
#             x, y = x.to(device), y.to(device)
#             lls = autoenc(x)
#             pred = torch.argmax(lls, dim=1)
#             acc += (pred == y).float().mean()
#         print(f"epoch {epoch} acc: {acc / len(valid_dl)}")

#     # show 1 reconstructed image
#     with torch.no_grad():
#         x, y = next(iter(valid_dl))
#         x, y = x.to(device), y.to(device)
#         lls = autoenc(x)
#         marginalized_scope = torch.randperm(128)[:32].tolist()
#         sample = autoenc.einet.sample(
#             evidence=autoenc.latent,
#             marginalized_scopes=marginalized_scope,
#             is_differentiable=True,
#             is_mpe=True,
#         )
#         recon_mpe = autoenc.decode(sample)
#         recon = autoenc.decode(autoenc.latent)
#         import matplotlib.pyplot as plt
#         import numpy as np

#         plt.imshow(np.transpose(x[0].cpu().numpy(), (1, 2, 0)))
#         plt.savefig("original.png")
#         plt.imshow(np.transpose(recon_mpe[0].cpu().numpy(), (1, 2, 0)))
#         plt.savefig("recon_mpe.png")
#         plt.imshow(np.transpose(recon[0].cpu().numpy(), (1, 2, 0)))
#         plt.savefig("recon.png")

#     print(f"epoch {epoch} loss: {loss.item()}")

autoenc.start_train(
    train_dl,
    valid_dl,
    device,
    learning_rate=0.0005,
    lambda_v=1.0,
    warmup_epochs=0,
    num_epochs=30,
)

# del train_dl, valid_dl
# print likelihoods of iD and OOD
autoenc.eval()
ll = 0
with torch.no_grad():
    for x, y in tqdm(test_dl):
        x, y = x.to(device), y.to(device)
        lls = autoenc(x)
        ll += lls.mean()
    print(f"ll id: {ll / len(test_dl)}")

    ll = 0
    for x, y in tqdm(ood_dl):
        x, y = x.to(device), y.to(device)
        lls = autoenc(x)
        ll += lls.mean()
    print(f"ll ood: {ll / len(ood_dl)}")
