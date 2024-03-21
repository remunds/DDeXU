import torch
from torchvision.models import efficientnet_v2_s
from torchvision import datasets, transforms
from torch.nn.utils.parametrizations import spectral_norm

# batchsize = 100  # 512
batchsize = 250  # 512
data_dir = "/data_docker/datasets/"
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

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

train_ds, valid_ds = torch.utils.data.random_split(
    train_ds, [0.8, 0.2], generator=torch.Generator().manual_seed(0)
)

train_dl = torch.utils.data.DataLoader(
    train_ds,
    batch_size=batchsize,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
    persistent_workers=True,
)
valid_dl = torch.utils.data.DataLoader(
    valid_ds,
    batch_size=batchsize,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
    persistent_workers=True,
)

from ResNetSPN import EfficientNetEnsemble

from ResNetSPN import EfficientNetDropout

from ResNetSPN import EfficientNetSPN


ckpt_dir = "ckpts/efficientnet_test/"
import os

os.makedirs(ckpt_dir, exist_ok=True)

train_params = dict(
    warmup_epochs=10,
    num_epochs=0,
    early_stop=10,
    lambda_v=1.0,
    deactivate_backbone=True,
    lr_schedule_warmup_step_size=10,
    lr_schedule_warmup_gamma=0.5,
    learning_rate_warmup=0.001,
    lr_schedule_step_size=10,
    lr_schedule_gamma=0.5,
    learning_rate=0.1,
)
model_params = dict(
    model="EfficientNetSPN",  # ConvResNetSPN, ConvResNetDDU
    block="basic",  # basic, bottleneck
    layers=[2, 2, 2, 2],
    num_classes=10,
    image_shape=(1, 28, 28),
    einet_depth=3,
    einet_num_sums=20,
    einet_num_leaves=20,
    einet_num_repetitions=1,
    einet_leaf_type="Normal",
    einet_dropout=0.0,
    spec_norm_bound=0.9,  # only for ConvResNetSPN
    spectral_normalization=True,  # only for ConvResNetDDU
    mod=True,  # only for ConvResNetDDU
)
# load model
model = EfficientNetEnsemble(**model_params)
# model = EfficientNetDropout(**model_params)
# model = EfficientNetSPN(**model_params)
# print(model)
model.to(device)
model.train()

train_params["deactivate_backbone"] = False
model.start_train(
    train_dl,
    valid_dl,
    device,
    checkpoint_dir=ckpt_dir,
    ensemble_num=2,
    **train_params,
)
