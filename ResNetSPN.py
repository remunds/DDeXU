import torch
import numpy as np
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
import torch.nn as nn
from spectral_normalization import spectral_norm
from simple_einet.einet import EinetConfig, Einet
from simple_einet.layers.distributions.normal import Normal


class EinetUtils:
    def start_train(self, dl, device, optimizer, lambda_v, num_epochs=1):
        self.train()
        for epoch in range(num_epochs):
            loss = 0.0
            for data, target in dl:
                optimizer.zero_grad()
                target = target.type(torch.LongTensor)
                data, target = data.to(device), target.to(device)
                output = self(data)
                loss_v = (
                    lambda_v * torch.nn.CrossEntropyLoss()(output, target)
                    + (1 - lambda_v) * -output.mean()
                )
                loss += loss_v.item()
                loss_v.backward()
                optimizer.step()
            print(f"Epoch {epoch}, loss {loss / len(dl.dataset)}")

    def eval_acc(self, dl, device):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in dl:
                data = data.to(device)
                labels = labels.to(device)
                total += labels.size(0)
                pred = self(data)
                pred = torch.argmax(pred, dim=1)
                correct += (pred == labels).sum().item()
        return correct / total

    def eval_ll(self, dl, device):
        self.eval()
        total = 0
        ll_total = 0
        with torch.no_grad():
            for data, labels in dl:
                data = data.to(device)
                labels = labels.to(device)
                total += labels.size(0)
                ll = self(data)
                ll_total += ll.mean().item()
        return ll_total / total

    def eval_pred_variance(self, dl, device):
        self.eval()
        pred_var_total = 0
        total = 0
        with torch.no_grad():
            for data, labels in dl:
                data = data.to(device)
                labels = labels.to(device)
                total += labels.size(0)
                pred_logit = self(data)
                pred = torch.softmax(pred_logit, dim=1)
                pred = torch.max(pred, dim=1)[0]
                pred_var = pred * (1 - pred)
                pred_var_total += pred_var.mean().item()
        return pred_var_total / total

    def eval_pred_entropy(self, dl, device):
        self.eval()
        pred_entropy_total = 0
        total = 0
        with torch.no_grad():
            for data, labels in dl:
                data = data.to(device)
                labels = labels.to(device)
                total += labels.size(0)
                pred_logit = self(data)
                pred = torch.softmax(pred_logit, dim=1)
                pred_entropy = -torch.sum(pred * torch.log(pred), dim=1)
                pred_entropy_total += pred_entropy.mean().item()
        return pred_entropy_total / total

    def explain_ll(self, dl, device):
        """
        Check each explaining variable individually.
        Returns the difference in log likelihood between the default model and the marginalized model.
        If largely positive, the variable explains the result.
        """
        ll_default = self.eval_ll(dl, device)
        explanations = []
        for i in self.explaining_vars:
            self.marginalized_scopes = [i]
            ll_marg = self.eval_ll(dl, device)
            explanations.append(ll_default - ll_marg)
        self.marginalized_scopes = None
        return explanations

    # def explain_mpe(self, dl, device):


class DenseResnet(nn.Module):
    """
    A simple fully connected ResNet.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_layers=3,
        num_hidden=128,
        dropout_rate=0.1,
        **classifier_kwargs,
    ):
        super(DenseResnet, self).__init__()
        # Defines class meta data.
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.classifier_kwargs = classifier_kwargs
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Defines the hidden layers.
        self.input_layer = nn.Linear(self.input_dim, self.num_hidden)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.dense_layers = [self.make_dense_layer() for _ in range(num_layers)]
        self.dense_layers = nn.ModuleList(self.dense_layers)

        # Defines the output layer.
        self.classifier = self.make_output_layer()

    def forward(self, inputs):
        # Projects the 2d input data to high dimension.
        hidden = self.input_layer(inputs)

        # Computes the ResNet hidden representations.
        for i in range(self.num_layers):
            resid = self.dense_layers[i](hidden)
            resid = self.dropout(resid)
            hidden = hidden + resid

        return self.classifier(hidden)

    def make_dense_layer(self):
        """Uses the Dense layer as the hidden layer."""
        return nn.Sequential(
            nn.Linear(self.num_hidden, self.num_hidden), self.activation
        )

    def make_output_layer(self):
        """Uses the Dense layer as the output layer."""
        return nn.Linear(self.num_hidden, self.output_dim, **self.classifier_kwargs)


class DenseResNetSPN(DenseResnet, EinetUtils):
    """
    Spectral normalized ResNet with einet as the output layer.
    """

    def __init__(self, spec_norm_bound=0.9, explaining_vars=[], **kwargs):
        self.spec_norm_bound = spec_norm_bound
        self.explaining_vars = explaining_vars
        super().__init__(**kwargs)

    def make_dense_layer(self):
        """applies spectral normalization to the hidden layer."""
        dense = nn.Linear(self.num_hidden, self.num_hidden)
        # todo: this is different to tf, since it does not use the spec_norm_bound...
        # note: both versions seem to work fine!
        # return nn.Sequential(
        #     nn.utils.parametrizations.spectral_norm(dense), self.activation
        # )
        return nn.Sequential(
            spectral_norm(dense, norm_bound=self.spec_norm_bound), self.activation
        )

    def make_output_layer(self):
        """uses einet as the output layer."""
        cfg = EinetConfig(
            num_features=self.num_hidden + len(self.explaining_vars),
            num_channels=1,
            depth=3,
            num_sums=20,
            num_leaves=20,
            num_repetitions=20,
            num_classes=self.output_dim,
            leaf_type=Normal,
            # leaf_kwargs={"total_count": 2**n_bits - 1},
            layer_type="einsum",
            dropout=0.0,
        )
        model = Einet(cfg)
        return model

    def forward(self, inputs):
        # extract explaining vars
        exp_vars = inputs[:, self.explaining_vars]
        # mask out explaining vars for resnet
        mask = torch.ones_like(inputs, dtype=torch.bool)
        mask[:, self.explaining_vars] = False
        inputs = inputs[mask]
        inputs = inputs.reshape(-1, self.input_dim)
        hidden = self.input_layer(inputs)

        # Computes the ResNet hidden representations.
        for i in range(self.num_layers):
            resid = self.dense_layers[i](hidden)
            resid = self.dropout(resid)
            hidden = hidden + resid

        # classifier is einet, so we need to concatenate the explaining vars
        hidden = torch.cat([hidden, exp_vars], dim=1)
        return self.classifier(hidden)


class ResidualBlockSN(BasicBlock):
    """
    Spectral normalized ResNet block.
    """

    def __init__(self, *args, **kwargs):
        spec_norm_bound = 6  # 0.9
        super(ResidualBlockSN, self).__init__(*args, **kwargs)
        # self.conv1 = nn.utils.spectral_norm(self.conv1)
        # self.conv2 = nn.utils.spectral_norm(self.conv2)
        self.conv1 = spectral_norm(self.conv1, norm_bound=spec_norm_bound)
        self.conv2 = spectral_norm(self.conv2, norm_bound=spec_norm_bound)


class BottleNeckSN(Bottleneck):
    """
    Spectral normalized ResNet block.
    """

    # def __init__(self, spec_norm_bound=0.9, *args, **kwargs):
    def __init__(self, *args, **kwargs):
        super(BottleNeckSN, self).__init__(*args, **kwargs)
        spec_norm_bound = 6  # 0.9
        # self.conv1 = nn.utils.spectral_norm(self.conv1)
        # self.conv2 = nn.utils.spectral_norm(self.conv2)
        # self.conv3 = nn.utils.spectral_norm(self.conv3)
        self.conv1 = spectral_norm(self.conv1, norm_bound=spec_norm_bound)
        self.conv2 = spectral_norm(self.conv2, norm_bound=spec_norm_bound)
        self.conv3 = spectral_norm(self.conv3, norm_bound=spec_norm_bound)


class ConvResNetSPN(ResNet, EinetUtils):
    """
    Spectral normalized convolutional ResNet with einet as the output layer.
    """

    def __init__(
        self,
        block,
        layers,
        num_classes,
        image_shape,  # (C, H, W)
        explaining_vars=[],  # indices of variables that should be explained
        spec_norm_bound=0.9,
        **kwargs,
    ):
        super(ConvResNetSPN, self).__init__(block, layers, num_classes, **kwargs)
        self.conv1 = nn.Conv2d(
            image_shape[0],
            64,  # self.inplanes,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        spec_norm_bound = 6
        # self.conv1 = nn.utils.spectral_norm(self.conv1)
        self.conv1 = spectral_norm(self.conv1, norm_bound=spec_norm_bound)
        self.fc = self.make_output_layer(
            512 * block.expansion + len(explaining_vars), num_classes
        )
        self.explaining_vars = explaining_vars
        self.image_shape = image_shape
        self.marginalized_scopes = None

    def make_output_layer(self, in_features, out_features):
        """Uses einet as the output layer."""
        cfg = EinetConfig(
            num_features=in_features,
            num_channels=1,
            depth=3,
            num_sums=20,
            num_leaves=20,
            num_repetitions=10,
            # num_repetitions=1,
            num_classes=out_features,
            leaf_type=Normal,
            # leaf_kwargs={"total_count": 2**n_bits - 1},
            layer_type="einsum",
            dropout=0.0,
        )
        model = Einet(cfg)
        return model

    def _forward_impl(self, x):
        # x is flattened
        # extract explaining vars
        exp_vars = x[:, self.explaining_vars]
        # mask out explaining vars for resnet
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:, self.explaining_vars] = False
        x = x[mask]
        # reshape to image
        x = x.reshape(-1, self.image_shape[0], self.image_shape[1], self.image_shape[2])

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
        x = self.fc(x, marginalized_scopes=self.marginalized_scopes)

        return x
