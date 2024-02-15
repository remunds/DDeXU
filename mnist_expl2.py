import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import os
import mlflow
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_datasets():
    data_dir = "/data_docker/datasets/"

    # load data
    tensor_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,)),
            # transforms.Lambda(lambda x: x.reshape(-1, 28 * 28).squeeze()),
        ]
    )
    mnist_transform = transforms.Compose(
        [
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.reshape(-1, 28 * 28).squeeze()),
        ]
    )

    # load mnist
    train_ds = datasets.MNIST(
        data_dir + "mnist",
        train=True,
        download=True,
        transform=tensor_transform,
    )
    valid_ds = datasets.MNIST(
        data_dir + "mnist",
        train=True,
        download=True,
        transform=tensor_transform,
    )

    test_ds = datasets.MNIST(
        data_dir + "mnist",
        train=False,
        download=True,
        transform=tensor_transform,
    )

    ood_ds = datasets.FashionMNIST(
        data_dir + "fashionmnist",
        train=False,
        download=True,
        transform=tensor_transform,
    )

    train_ds, valid_ds = torch.utils.data.random_split(
        train_ds, [0.8, 0.2], generator=torch.Generator().manual_seed(0)
    )

    return (train_ds, valid_ds, test_ds, ood_ds, mnist_transform)


train_ds, valid_ds, test_ds, ood_ds, mnist_transform = get_datasets()
print(len(train_ds))

# most examples in train have between 100 and 200 pixels != 0
train_between_100_200 = []
train_smaller = []
train_larger = []
for img, target in train_ds:
    non_zero = img.count_nonzero()
    img = mnist_transform(img)
    # concatenate non_zero to beginning of img
    img = np.concatenate((np.array([non_zero]), img.numpy())).astype(np.float32)
    if 100 < non_zero < 200:
        train_between_100_200.append((img, target))
    elif non_zero < 100:
        train_smaller.append((img, target))
    else:
        train_larger.append((img, target))
print(len(train_between_100_200))

inside_train, inside_val = torch.utils.data.random_split(
    train_between_100_200, [0.8, 0.2], generator=torch.Generator().manual_seed(0)
)

train_dl = DataLoader(
    inside_train,
    batch_size=512,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

valid_dl = DataLoader(
    inside_val,
    batch_size=512,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
)


from ResNetSPN import EfficientNetSPN

model = EfficientNetSPN(
    num_classes=10,
    image_shape=(1, 28, 28),
    explaining_vars=[0],
    einet_depth=5,
    einet_num_sums=15,
    einet_num_leaves=20,
    einet_num_repetitions=25,
    einet_leaf_type="Normal",
    einet_dropout=0.0,
)

ckpt_dir = f"/data_docker/ckpts/mnist-expl2/"
train_params = dict(
    pretrained_path=None,
    learning_rate_warmup=0.05,
    learning_rate=0.07,
    num_epochs=0,
    warmup_epochs=10,
    early_stop=10,
    lambda_v=0.5,
    deactivate_backbone=True,
    # use_mpe_reconstruction_loss=True,
)

model.to(device)

model.start_train(
    train_dl,
    valid_dl,
    device,
    checkpoint_dir=ckpt_dir,
    trial=None,
    **train_params,
)

model.eval()

embeddings = model.get_embeddings(valid_dl, device)

from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType

print("embeddings shape: ", embeddings.shape)
meta_types = [MetaType.REAL for _ in range(embeddings.shape[1])]
meta_types[0] = MetaType.DISCRETE  # expl. var

ds_context = Context(meta_types=meta_types)
train_data = embeddings.cpu().detach().numpy()[:1000, :]
train_data = embeddings[:300, :]
ds_context.add_domains(train_data)

from spn.algorithms.LearningWrappers import learn_mspn

print("learning mspn")
mspn = learn_mspn(train_data, ds_context, min_instances_slice=20)


from spn.algorithms.MPE import mpe

mpe_data = train_data[:5].copy()
mpe_data[:, 0] = np.nan
print("train: ", train_data[:5, 0])
print("mpe: ", mpe(spn, mpe_data)[:5, 0])

mpe_data = embeddings[1000:1100, :].cpu().detach().numpy()
print("actual: ", mpe_data[:5, 0])
mpe_data[:, 0] = np.nan
mpe2 = mpe(spn, mpe_data)
print("mpe: ", mpe2[:5, 0])

print(
    "difference: ",
    np.abs(mpe2[:, 0] - embeddings[1000:1100, 0].cpu().detach().numpy()).mean(),
)


# model.deactivate_uncert_head()
# backbone_valid_acc = model.eval_acc(valid_dl, device)
# print(f"Backbone validation accuracy: {backbone_valid_acc}")
# model.activate_uncert_head()


# valid_acc = model.eval_acc(valid_dl, device)
# print(f"Validation accuracy: {valid_acc}")

# valid_ll = model.eval_ll_marg(None, device, valid_dl)
# print(f"Validation marginal log likelihood: {valid_ll}")

# smaller_dl = DataLoader(
#     train_smaller,
#     batch_size=512,
#     shuffle=True,
#     num_workers=2,
#     pin_memory=True,
# )

# larger_dl = DataLoader(
#     train_larger,
#     batch_size=512,
#     shuffle=True,
#     num_workers=2,
#     pin_memory=True,
# )

# smaller_acc = model.eval_acc(smaller_dl, device)
# print(f"smaller accuracy: {smaller_acc}")

# smaller_ll = model.eval_ll_marg(None, device, smaller_dl)
# print(f"smaller marginal log likelihood: {smaller_ll}")

# larger_acc = model.eval_acc(larger_dl, device)
# print(f"larger accuracy: {larger_acc}")

# larger_ll = model.eval_ll_marg(None, device, larger_dl)
# print(f"larger marginal log likelihood: {larger_ll}")

# ll_expl_valid = model.explain_ll(valid_dl, device)
# total = 0
# for ll in ll_expl_valid:
#     total += ll
# print(f"Explained log likelihood valid: {total}")

# ll_expl_smaller = model.explain_ll(smaller_dl, device)
# total = 0
# for ll in ll_expl_smaller:
#     total += ll
# print(f"Explained log likelihood smaller: {total}")

# ll_expl_larger = model.explain_ll(larger_dl, device)
# total = 0
# for ll in ll_expl_larger:
#     total += ll
# print(f"Explained log likelihood larger: {total}")

# mpe_valid = model.explain_mpe(valid_dl, device, return_all=True)
# mpe_valid = torch.concat([i.flatten() for i in mpe_valid]).cpu().detach().numpy()
# print(f"First 20 MPE valid: {mpe_valid[:20]}")
# really_valid = np.array([i[0][0] for i in inside_val])
# differences = np.abs(mpe_valid - really_valid).mean()
# print(f"Mean absolute difference valid: {differences}")

# mpe_smaller = model.explain_mpe(smaller_dl, device, return_all=True)
# mpe_smaller = torch.concat([i.flatten() for i in mpe_smaller]).cpu().detach().numpy()
# print(f"First 20 MPE smaller: {mpe_smaller[:20]}")
# really_smaller = np.array([i[0][0] for i in train_smaller])
# differences = np.abs(mpe_smaller - really_smaller).mean()
# print(f"Mean absolute difference smaller: {differences}")

# mpe_larger = model.explain_mpe(larger_dl, device, return_all=True)
# mpe_larger = torch.concat([i.flatten() for i in mpe_larger]).cpu().detach().numpy()
# print(f"First 20 MPE larger: {mpe_larger[:20]}")
# really_larger = np.array([i[0][0] for i in train_larger])
# differences = np.abs(mpe_larger - really_larger).mean()
# print(f"Mean absolute difference larger: {differences}")
