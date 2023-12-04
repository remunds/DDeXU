import torch
import numpy as np
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
import torch.nn as nn
from spectral_normalization import spectral_norm
from simple_einet.einet import EinetConfig, Einet
from simple_einet.layers.distributions.normal import Normal


class ConvResNet(ResNet):
    """
    ResNet model with ability to return latent space
    """

    def __init__(self, in_channels, *args, **kwargs):
        super(ConvResNet, self).__init__(*args, **kwargs)
        # adapt for mnist -> channel=1
        self.conv1 = torch.nn.Conv2d(
            in_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def get_hidden(self, x):
        for name, module in self.named_children():
            if name != "fc":
                x = module(x)
        return x


def resnet_from_path(path):
    resnet = ConvResNet(1, BasicBlock, [2, 2, 2, 2], num_classes=10)
    resnet.load(path)
    return resnet


def train_eval_resnet(num_classes, train_dl, test_dl, device, save_dir):
    resnet = ConvResNet(1, BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

    resnet.to(device)

    # train NN
    optimizer = torch.optim.Adam(resnet.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    print("start training resnet")

    for epoch in range(3):
        for data, target in train_dl:
            optimizer.zero_grad()

            data, target = data.to(device), target.to(device)
            output = resnet(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, loss {loss.item()}")

    # evaluate NN
    resnet.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_dl:
            data, target = data.to(device), target.to(device)
            output = resnet(data)
            loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        loss /= len(test_dl)
        print(f"Test loss: {loss}")
        print(f"Test accuracy: {correct / len(test_dl.dataset)}")

    torch.save(resnet.state_dict(), f"{save_dir}/resnet.pt")
    return resnet


def get_latent_dataset(
    train_ds, train_dl, test_ds, test_dl, resnet, device, batchsize_resnet, save_dir
):
    latent_train = np.zeros((train_ds.data.shape[0], 512))
    target_train = np.zeros((train_ds.data.shape[0],))
    latent_test = np.zeros((test_ds.data.shape[0], 512))
    target_test = np.zeros((test_ds.data.shape[0],))
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_dl):
            data = data.to(device)
            target = target.to(device)
            low_idx = batch_idx * batchsize_resnet
            high_idx = (batch_idx + 1) * batchsize_resnet
            latent_train[low_idx:high_idx] = (
                resnet.get_hidden(data).squeeze().cpu().numpy()
            )
            target_train[low_idx:high_idx] = target.cpu().numpy()
        np.save(f"{save_dir}/latent_train.npy", latent_train)
        np.save(f"{save_dir}/target_train.npy", target_train)
        print("Latent train dataset shape: ", latent_train.shape)

        for batch_idx, (data, target) in enumerate(test_dl):
            data = data.to(device)
            target = target.to(device)
            low_idx = batch_idx * batchsize_resnet
            high_idx = (batch_idx + 1) * batchsize_resnet
            latent_test[low_idx:high_idx] = (
                resnet.get_hidden(data).squeeze().cpu().numpy()
            )
            target_test[low_idx:high_idx] = target.cpu().numpy()
        np.save(f"{save_dir}/latent_test.npy", latent_test)
        np.save(f"{save_dir}/target_test.npy", target_test)
    print("done collecting latent space dataset")
    return latent_train, target_train, latent_test, target_test


def get_latent_batched(data_loader, size, resnet, device, batchsize_resnet, save_dir):
    latent = np.zeros((size, 512))
    latent_target = np.zeros((size,))
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(device)
            target = target.to(device)
            low_idx = batch_idx * batchsize_resnet
            high_idx = (batch_idx + 1) * batchsize_resnet
            latent[low_idx:high_idx] = resnet.get_hidden(data).squeeze().cpu().numpy()
            latent_target[low_idx:high_idx] = target.cpu().numpy()
    return latent, latent_target


def train_small_mlp(
    latent_train, target_train, latent_test, target_test, device, batchsize_resnet
):
    # train simple MLP on latent space for mnist classification to show that latent space is useful

    mlp = nn.Sequential(
        nn.Linear(512, 100), nn.ReLU(), nn.Linear(100, 10), nn.LogSoftmax(dim=1)
    )

    mlp.to(device)

    # train MLP
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    latent_train = torch.from_numpy(latent_train).to(dtype=torch.float32)
    target_train = torch.from_numpy(target_train).to(dtype=torch.long)
    latent_test = torch.from_numpy(latent_test).to(dtype=torch.float32)
    target_test = torch.from_numpy(target_test).to(dtype=torch.long)

    print("start training mlp")
    for epoch in range(2):
        for batch_idx in range(0, latent_train.shape[0], batchsize_resnet):
            optimizer.zero_grad()

            data = latent_train[batch_idx : batch_idx + batchsize_resnet]
            target = target_train[batch_idx : batch_idx + batchsize_resnet]
            data, target = data.to(device), target.to(device)
            output = mlp(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, loss {loss.item()}")

    # evaluate MLP
    mlp.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx in range(0, latent_test.shape[0], batchsize_resnet):
            data = latent_test[batch_idx : batch_idx + batchsize_resnet]
            target = target_test[batch_idx : batch_idx + batchsize_resnet]
            data, target = data.to(device), target.to(device)
            output = mlp(data)
            loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        loss /= latent_test.shape[0] / batchsize_resnet
        print(f"Test loss: {loss}")
        print(f"Test accuracy: {correct / latent_test.shape[0]}")


class ResidualBlockSN(BasicBlock):
    def __init__(self, *args, **kwargs):
        spec_norm_bound = 0.9
        super(ResidualBlockSN, self).__init__(*args, **kwargs)
        # self.conv1 = nn.utils.spectral_norm(self.conv1)
        # self.conv2 = nn.utils.spectral_norm(self.conv2)
        self.conv1 = spectral_norm(self.conv1, norm_bound=spec_norm_bound)
        self.conv2 = spectral_norm(self.conv2, norm_bound=spec_norm_bound)


class BottleNeckSN(Bottleneck):
    # def __init__(self, spec_norm_bound=0.9, *args, **kwargs):
    def __init__(self, *args, **kwargs):
        super(BottleNeckSN, self).__init__(*args, **kwargs)
        spec_norm_bound = 0.9
        # self.conv1 = nn.utils.spectral_norm(self.conv1)
        # self.conv2 = nn.utils.spectral_norm(self.conv2)
        # self.conv3 = nn.utils.spectral_norm(self.conv3)
        self.conv1 = spectral_norm(self.conv1, norm_bound=spec_norm_bound)
        self.conv2 = spectral_norm(self.conv2, norm_bound=spec_norm_bound)
        self.conv3 = spectral_norm(self.conv3, norm_bound=spec_norm_bound)


class ResNetSPN(ResNet):
    def __init__(
        self,
        block,
        layers,
        num_classes,
        explaining_vars,  # indices of variables that should be explained
        spec_norm_bound=0.9,
        num_channels=1,
        **kwargs,
    ):
        super(ResNetSPN, self).__init__(block, layers, num_classes, **kwargs)
        self.conv1 = nn.Conv2d(
            num_channels,
            64,  # self.inplanes,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        # self.conv1 = nn.utils.spectral_norm(self.conv1)
        self.conv1 = spectral_norm(self.conv1, norm_bound=spec_norm_bound)
        self.fc = self.make_output_layer(
            512 * block.expansion + len(explaining_vars), num_classes
        )
        self.explaining_vars = explaining_vars

    def make_output_layer(self, in_features, out_features):
        """Uses einet as the output layer."""
        cfg = EinetConfig(
            num_features=in_features,
            num_channels=1,
            depth=3,
            num_sums=20,
            num_leaves=20,
            # num_repetitions=20,
            num_repetitions=1,
            num_classes=out_features,
            leaf_type=Normal,
            # leaf_kwargs={"total_count": 2**n_bits - 1},
            layer_type="einsum",
            dropout=0.0,
        )
        model = Einet(cfg)
        return model

    def _forward_impl(self, x):
        exp_vars = x[:, self.explaining_vars]
        # mask out explaining vars for resnet
        mask = torch.ones_like(x)
        mask[:, self.explaining_vars] = 0
        x = x * mask

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # fc is einet, so we need to concatenate the explaining vars
        x = torch.cat([x, exp_vars], dim=1)
        x = self.fc(x)

        return x
