import os
from typing import Any
import torch

from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

# from net.resnet import ResNet, BasicBlock, Bottleneck
import torch.nn as nn
from spectral_normalization import spectral_norm
from simple_einet.einet import EinetConfig, Einet

from simple_einet.layers.distributions.normal import Normal
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import mlflow
import optuna
import matplotlib.pyplot as plt


class EinetUtils:
    def start_train(
        self,
        dl_train,
        dl_valid,
        device,
        learning_rate_warmup,
        learning_rate,
        lambda_v,
        warmup_epochs,
        num_epochs,
        deactivate_backbone=True,
        lr_schedule_warmup_step_size=None,
        lr_schedule_warmup_gamma=None,
        lr_schedule_step_size=None,
        lr_schedule_gamma=None,
        early_stop=3,
        checkpoint_dir=None,
        trial=None,
    ):
        self.train()
        self.deactivate_einet()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate_warmup)
        if (
            lr_schedule_warmup_step_size is not None
            and lr_schedule_warmup_gamma is not None
        ):
            lr_schedule_resnet = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=lr_schedule_warmup_step_size,
                gamma=lr_schedule_warmup_gamma,
            )

        val_increase = 0
        lowest_val_loss = torch.inf
        epoch = -1
        # warmup by only training backbone
        t = tqdm(range(warmup_epochs))
        for epoch in t:
            t.set_description(f"Epoch {epoch}")
            loss = 0.0
            for data, target in dl_train:
                optimizer.zero_grad()
                target = target.type(torch.LongTensor)
                data, target = data.to(device), target.to(device)
                output = self(data)
                loss_v = torch.nn.CrossEntropyLoss()(output, target)
                loss += loss_v.item()
                loss_v.backward()
                optimizer.step()
            if lr_schedule_resnet:
                lr_schedule_resnet.step()

            val_loss = 0.0
            with torch.no_grad():
                for data, target in dl_valid:
                    optimizer.zero_grad()
                    target = target.type(torch.LongTensor)
                    data, target = data.to(device), target.to(device)
                    output = self(data)
                    loss_v = torch.nn.CrossEntropyLoss()(output, target)
                    val_loss += loss_v.item()
            t.set_postfix(
                dict(
                    train_loss=loss / len(dl_train.dataset),
                    val_loss=val_loss / len(dl_valid.dataset),
                )
            )
            mlflow.log_metric(
                key="train_loss", value=loss / len(dl_train.dataset), step=epoch
            )
            mlflow.log_metric(
                key="val_loss", value=val_loss / len(dl_valid.dataset), step=epoch
            )
            # early stopping via optuna
            if trial:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    mlflow.set_tag("pruned", f"backbone: {epoch}")
                    raise optuna.TrialPruned()
            if val_loss <= lowest_val_loss:
                lowest_val_loss = val_loss
                val_increase = 0
                if checkpoint_dir:
                    self.save(checkpoint_dir + "checkpoint.pt")
            else:
                # early stopping when val increases
                val_increase += 1
                if val_increase >= early_stop:
                    print(
                        f"Stopping Backbone early, val loss increased for the last {early_stop} epochs."
                    )
                    break

        warmup_epochs_performed = epoch + 1
        # load best
        if checkpoint_dir and warmup_epochs_performed > 0:
            self.load(checkpoint_dir + "checkpoint.pt")

        # train einet (and optionally resnet jointly)
        self.activate_einet(deactivate_backbone)
        if deactivate_backbone:
            optimizer = torch.optim.Adam(self.einet.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        if lr_schedule_step_size is not None and lr_schedule_gamma is not None:
            lr_schedule_einet = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=lr_schedule_step_size,
                gamma=lr_schedule_gamma,
            )
        lowest_val_loss = torch.inf
        val_increase = 0
        t = tqdm(range(num_epochs))
        for epoch in t:
            t.set_description(f"Epoch {warmup_epochs_performed + epoch}")
            loss = 0.0
            for data, target in dl_train:
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
            if lr_schedule_einet:
                lr_schedule_einet.step()

            val_loss = 0.0
            with torch.no_grad():
                for data, target in dl_valid:
                    optimizer.zero_grad()
                    target = target.type(torch.LongTensor)
                    data, target = data.to(device), target.to(device)
                    output = self(data)
                    loss_v = (
                        lambda_v * torch.nn.CrossEntropyLoss()(output, target)
                        + (1 - lambda_v) * -output.mean()
                    )
                    val_loss += loss_v.item()
                t.set_postfix(
                    dict(
                        train_loss=loss / len(dl_train.dataset),
                        val_loss=val_loss / len(dl_valid.dataset),
                    )
                )
                mlflow.log_metric(
                    key="train_loss",
                    value=loss / len(dl_train.dataset),
                    step=warmup_epochs_performed + epoch,
                )
                mlflow.log_metric(
                    key="val_loss",
                    value=val_loss / len(dl_valid.dataset),
                    step=warmup_epochs_performed + epoch,
                )
                # early stopping via optuna
                if trial:
                    trial.report(val_loss, warmup_epochs + epoch)
                    if trial.should_prune():
                        mlflow.set_tag("pruned", f"einet: {epoch}")
                        raise optuna.TrialPruned()
                if val_loss <= lowest_val_loss:
                    lowest_val_loss = val_loss
                    val_increase = 0
                    if checkpoint_dir:
                        self.save(checkpoint_dir + "checkpoint.pt")
                else:
                    # early stopping when val increases
                    val_increase += 1
                    if val_increase >= early_stop:
                        print(
                            f"Stopping Einet early, val loss increased for the last {early_stop} epochs."
                        )
                        break
        # load best (with einet active)
        if checkpoint_dir is not None:
            self.load(checkpoint_dir + "checkpoint.pt")
        return lowest_val_loss

    def save(self, path):
        # create path if not exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path, backbone_only=False):
        self.load_state_dict(torch.load(path))
        if backbone_only:
            # reinitialize einet
            for param in self.einet.parameters():
                torch.nn.init.normal_(param, mean=0.0, std=0.01)

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

    def eval_ll(self, dl, device, return_all=False):
        self.eval()
        index = 0
        lls = torch.zeros(len(dl.dataset))
        with torch.no_grad():
            for data, _ in dl:
                data = data.to(device)
                ll = self(data)
                lls[index : index + len(ll)] = ll.mean(dim=1)
                index += len(ll)
        if return_all:
            return lls
        return torch.mean(lls).item()

    def eval_pred_variance(self, dl, device, return_all=False):
        self.eval()
        index = 0
        pred_vars = torch.zeros(len(dl.dataset))
        with torch.no_grad():
            for data, _ in dl:
                data = data.to(device)
                pred_logit = self(data)
                pred = torch.softmax(pred_logit, dim=1)
                pred = torch.max(pred, dim=1)[0]
                pred_var = pred * (1 - pred)
                pred_vars[index : index + len(pred_var)] = pred_var
                index += len(pred_var)
        if return_all:
            return pred_vars
        return torch.mean(pred_vars).item()

    def eval_pred_entropy(self, dl, device, return_all=False):
        self.eval()
        index = 0
        pred_entropies = torch.zeros(len(dl.dataset))
        with torch.no_grad():
            for data, _ in dl:
                data = data.to(device)
                pred_logit = self(data)
                pred = torch.softmax(pred_logit, dim=1)
                pred_entropy = -torch.sum(pred * torch.log(pred), dim=1)
                pred_entropies[index : index + len(pred_entropy)] = pred_entropy
                index += len(pred_entropy)
        if return_all:
            return pred_entropies
        return torch.mean(pred_entropies).item()

    def eval_dempster_shafer(self, dl, device, return_all=False):
        """
        SNGP: better for distance aware models, where
        magnitude of logits reflects distance from observed data manifold
        """
        num_classes = self.einet.config.num_classes
        self.eval()
        index = 0
        uncertainties = torch.zeros(len(dl.dataset))
        with torch.no_grad():
            for data in dl:
                data = data.to(device)
                lls = self(data)
                # assumes negative log likelihoods
                # not really sure if normalizing even makes sense
                # but without, values are always 1(-) or 0(+)
                # lls_norm = lls / torch.min(lls, dim=1)[0].unsqueeze(1)
                lls_norm = lls - torch.mean(lls, dim=1).unsqueeze(1)
                lls_norm = lls_norm / torch.max(torch.abs(lls_norm), dim=1)[
                    0
                ].unsqueeze(1)
                uncertainty = num_classes / (
                    num_classes + torch.sum(torch.exp(lls_norm), dim=1)
                )
                uncertainties[index : index + len(uncertainty)] = uncertainty
                index += len(uncertainty)
        if return_all:
            return uncertainties
        return torch.mean(uncertainties).item()

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
            explanations.append((ll_marg - ll_default))
        self.marginalized_scopes = None
        return explanations

    def explain_mpe(self, dl, device):
        expl_var_vals = []
        expl_var_mpes = []
        for data, _ in dl:
            data = data.to(device)
            # extract explaining vars
            exp_vars = data[:, self.explaining_vars]
            expl_var_vals.append(exp_vars)
            # mask out explaining vars for backbone
            mask = torch.ones_like(data, dtype=torch.bool)
            mask[:, self.explaining_vars] = False
            data = data[mask]
            if self.image_shape is not None:
                # ConvResNetSPN
                # reshape to image
                data = data.reshape(
                    -1, self.image_shape[0], self.image_shape[1], self.image_shape[2]
                )
            else:
                # DenseResNetSPN
                data = data.reshape(-1, self.input_dim)

            # extract most probable explanation of current input
            hidden = self.forward_hidden(data)
            hidden = torch.cat([hidden, exp_vars], dim=1)
            mpe = self.einet.mpe(
                evidence=hidden, marginalized_scopes=self.explaining_vars
            )
            expl_var_mpes.append(mpe[:, self.explaining_vars])
        expl_var_vals = torch.cat(expl_var_vals, dim=0)
        expl_var_mpes = torch.cat(expl_var_mpes, dim=0)
        return torch.abs(expl_var_mpes - expl_var_vals).mean(dim=0).cpu().numpy()


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

    def __init__(
        self,
        spec_norm_bound=0.9,
        explaining_vars=[],
        einet_depth=3,
        einet_num_sums=20,
        einet_num_leaves=20,
        einet_num_repetitions=1,
        einet_leaf_type="Normal",
        einet_dropout=0.0,
        **kwargs,
    ):
        self.spec_norm_bound = spec_norm_bound
        self.explaining_vars = explaining_vars
        self.einet_depth = einet_depth
        self.einet_num_sums = einet_num_sums
        self.einet_num_leaves = einet_num_leaves
        self.einet_num_repetitions = einet_num_repetitions
        if einet_leaf_type == "Normal":
            self.einet_leaf_type = Normal
        else:
            raise NotImplementedError
        self.einet_dropout = einet_dropout

        super().__init__(**kwargs)
        self.einet = self.make_einet_output_layer()

    def activate_einet(self, deactivate_backbone=True):
        """
        Activates the einet output layer for second stage training and inference.
        """
        self.einet_active = True
        if deactivate_backbone:
            for param in self.parameters():
                param.requires_grad = False
        else:
            for param in self.parameters():
                param.requires_grad = True
        for param in self.einet.parameters():
            param.requires_grad = True

    def deactivate_einet(self):
        """
        Deactivates the einet output layer for first stage training.
        """
        self.einet_active = False
        for param in self.einet.parameters():
            param.requires_grad = False

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

    def make_einet_output_layer(self):
        """uses einet as the output layer."""
        cfg = EinetConfig(
            num_features=self.num_hidden + len(self.explaining_vars),
            num_channels=1,
            depth=self.einet_depth,
            num_sums=self.einet_num_sums,
            num_leaves=self.einet_num_leaves,
            num_repetitions=self.einet_num_repetitions,
            num_classes=self.output_dim,
            leaf_type=self.einet_leaf_type,
            # leaf_kwargs={"total_count": 2**n_bits - 1},
            layer_type="einsum",
            dropout=self.einet_dropout,
        )
        model = Einet(cfg)
        return model

    def forward_hidden(self, inputs):
        hidden = self.input_layer(inputs)

        # Computes the ResNet hidden representations.
        for i in range(self.num_layers):
            resid = self.dense_layers[i](hidden)
            resid = self.dropout(resid)
            hidden = hidden + resid

        return hidden

    def forward(self, inputs):
        # extract explaining vars
        exp_vars = inputs[:, self.explaining_vars]
        # mask out explaining vars for bakcbone
        mask = torch.ones_like(inputs, dtype=torch.bool)
        mask[:, self.explaining_vars] = False
        inputs = inputs[mask]
        inputs = inputs.reshape(-1, self.input_dim)
        # feed through resnet
        hidden = self.forward_hidden(inputs)

        if self.einet_active:
            # classifier is einet, so we need to concatenate the explaining vars
            hidden = torch.cat([hidden, exp_vars], dim=1)
            return self.einet(hidden)

        return self.classifier(hidden)


class ResidualBlockSN(BasicBlock):
    """
    Spectral normalized ResNet block.
    """

    def __init__(self, *args, spec_norm_bound=0.9, **kwargs):
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
    def __init__(self, *args, spec_norm_bound=0.9, **kwargs):
        super(BottleNeckSN, self).__init__(*args, **kwargs)
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
        einet_depth=3,
        einet_num_sums=20,
        einet_num_leaves=20,
        einet_num_repetitions=1,
        einet_leaf_type="Normal",
        einet_dropout=0.0,
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
        # self.conv1 = nn.utils.spectral_norm(self.conv1)
        self.conv1 = spectral_norm(self.conv1, norm_bound=spec_norm_bound)
        self.explaining_vars = explaining_vars
        self.einet_depth = einet_depth
        self.einet_num_sums = einet_num_sums
        self.einet_num_leaves = einet_num_leaves
        self.einet_num_repetitions = einet_num_repetitions
        if einet_leaf_type == "Normal":
            self.einet_leaf_type = Normal
        else:
            raise NotImplementedError
        self.einet_dropout = einet_dropout
        self.image_shape = image_shape
        self.marginalized_scopes = None
        self.einet = self.make_einet_output_layer(
            512 * block.expansion + len(explaining_vars), num_classes
        )

    def activate_einet(self, deactivate_backbone=True):
        """
        Activates the einet output layer for second stage training and inference.
        """
        self.einet_active = True
        if deactivate_backbone:
            for param in self.parameters():
                param.requires_grad = False
        else:
            for param in self.parameters():
                param.requires_grad = True
        for param in self.einet.parameters():
            param.requires_grad = True

    def deactivate_einet(self):
        """
        Deactivates the einet output layer for first stage training.
        """
        self.einet_active = False
        for param in self.einet.parameters():
            param.requires_grad = False

    def make_einet_output_layer(self, in_features, out_features):
        """Uses einet as the output layer."""
        cfg = EinetConfig(
            num_features=in_features,
            num_channels=1,
            depth=self.einet_depth,
            num_sums=self.einet_num_sums,
            num_leaves=self.einet_num_leaves,
            num_repetitions=self.einet_num_repetitions,
            num_classes=out_features,
            leaf_type=self.einet_leaf_type,
            layer_type="einsum",
            dropout=self.einet_dropout,
        )
        model = Einet(cfg)
        return model

    def forward_hidden(self, x):
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
        return x

    def _forward_impl(self, x):
        # x is flattened
        # extract explaining vars
        exp_vars = x[:, self.explaining_vars]
        # mask out explaining vars for backbone
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:, self.explaining_vars] = False
        x = x[mask]
        # reshape to image
        x = x.reshape(-1, self.image_shape[0], self.image_shape[1], self.image_shape[2])

        # feed through resnet
        x = self.forward_hidden(x)

        if self.einet_active:
            # classifier is einet, so we need to concatenate the explaining vars
            x = torch.cat([x, exp_vars], dim=1)
            return self.einet(x, marginalized_scopes=self.marginalized_scopes)

        return self.fc(x)


from net.resnet import ResNet


class ConvResnetDDU(ResNet, EinetUtils):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes=10,
        temp=1.0,
        spectral_normalization=True,
        mod=True,
        coeff=3,
        n_power_iterations=1,
        image_shape=(1, 28, 28),  # (C, H, W)
        explaining_vars=[],  # indices of variables that should be explained
        einet_depth=3,
        einet_num_sums=20,
        einet_num_leaves=20,
        einet_num_repetitions=1,
        einet_leaf_type="Normal",
        einet_dropout=0.0,
        **kwargs,
    ):
        mnist = image_shape[0] == 1 and image_shape[1] == 28 and image_shape[2] == 28
        super(ConvResnetDDU, self).__init__(
            block,
            num_blocks,
            num_classes,
            temp,
            spectral_normalization,
            mod,
            coeff,
            n_power_iterations,
            mnist,
        )

        self.explaining_vars = explaining_vars
        self.image_shape = image_shape
        self.marginalized_scopes = None
        self.einet_depth = einet_depth
        self.einet_num_sums = einet_num_sums
        self.einet_num_leaves = einet_num_leaves
        self.einet_num_repetitions = einet_num_repetitions
        if einet_leaf_type == "Normal":
            self.einet_leaf_type = Normal
        else:
            raise NotImplementedError
        self.einet_dropout = einet_dropout
        self.einet = self.make_einet_output_layer(
            512 * block.expansion + len(explaining_vars), num_classes
        )

    def activate_einet(self, deactivate_backbone=True):
        """
        Activates the einet output layer for second stage training and inference.
        """
        self.einet_active = True
        if deactivate_backbone:
            for param in self.parameters():
                param.requires_grad = False
        else:
            for param in self.parameters():
                param.requires_grad = True
        for param in self.einet.parameters():
            param.requires_grad = True

    def deactivate_einet(self):
        """
        Deactivates the einet output layer for first stage training.
        """
        self.einet_active = False
        for param in self.einet.parameters():
            param.requires_grad = False

    def make_einet_output_layer(self, in_features, out_features):
        """Uses einet as the output layer."""
        cfg = EinetConfig(
            num_features=in_features,
            num_channels=1,
            depth=self.einet_depth,
            num_sums=self.einet_num_sums,
            num_leaves=self.einet_num_leaves,
            num_repetitions=self.einet_num_repetitions,
            num_classes=out_features,
            leaf_type=self.einet_leaf_type,
            layer_type="einsum",
            dropout=self.einet_dropout,
        )
        model = Einet(cfg)
        return model

    def forward_hidden(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        # x is flattened
        # extract explaining vars
        exp_vars = x[:, self.explaining_vars]
        # mask out explaining vars for resnet
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:, self.explaining_vars] = False
        x = x[mask]
        # reshape to image
        x = x.reshape(-1, self.image_shape[0], self.image_shape[1], self.image_shape[2])

        # feed through resnet
        x = self.forward_hidden(x)

        if self.einet_active:
            # classifier is einet, so we need to concatenate the explaining vars
            x = torch.cat([x, exp_vars], dim=1)
            return self.einet(x, marginalized_scopes=self.marginalized_scopes)

        return self.fc(x) / self.temp


class AutoEncoderSPN(nn.Module, EinetUtils):
    """
    Performs some Convolutions to get latent embeddings, then trains SPN on those for downstream tasks.
    Lastly, uses stochastically conditioned MPE of SPN and DeConvolution to reconstruct the original image.
    """

    def __init__(
        self,
        explaining_vars=[],
        num_classes=10,
        image_shape=(1, 28, 28),  # (C, H, W)
        einet_depth=3,
        einet_num_sums=20,
        einet_num_leaves=20,
        einet_num_repetitions=1,
        einet_leaf_type="Normal",
        einet_dropout=0.0,
        marginalized_scopes=None,
        **kwargs,
    ):
        super(AutoEncoderSPN, self).__init__()
        self.explaining_vars = explaining_vars
        self.image_shape = image_shape
        self.marginalized_scopes = marginalized_scopes
        self.einet_depth = einet_depth
        self.einet_num_sums = einet_num_sums
        self.einet_num_leaves = einet_num_leaves
        self.einet_num_repetitions = einet_num_repetitions
        if einet_leaf_type == "Normal":
            self.einet_leaf_type = Normal
        else:
            raise NotImplementedError
        self.einet_dropout = einet_dropout

        # specify encoder
        self.conv1 = nn.Conv2d(image_shape[0], 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.activation = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128 * image_shape[1] * image_shape[2], 128)

        # specify decoder
        self.fc2 = nn.Linear(128, 128 * image_shape[1] * image_shape[2])
        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, image_shape[0], 3, padding=1)
        self.latent = None

        self.einet = self.make_einet_layer(128 + len(explaining_vars), num_classes)

    def make_einet_layer(self, in_features, out_features):
        """Uses einet as the output layer."""
        cfg = EinetConfig(
            num_features=in_features,
            num_channels=1,
            depth=self.einet_depth,
            num_sums=self.einet_num_sums,
            num_leaves=self.einet_num_leaves,
            num_repetitions=self.einet_num_repetitions,
            num_classes=out_features,
            leaf_type=Normal,
            layer_type="einsum",
            dropout=self.einet_dropout,
        )
        model = Einet(cfg)
        return model

    def forward(self, x):
        # x is flattened
        # extract explaining vars
        exp_vars = x[:, self.explaining_vars]
        # mask out explaining vars for backbone
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:, self.explaining_vars] = False
        x = x[mask]
        # reshape to image
        x = x.reshape(-1, self.image_shape[0], self.image_shape[1], self.image_shape[2])

        # encode
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = x.view(-1, 128 * self.image_shape[1] * self.image_shape[2])
        x = self.fc(x)
        self.latent = x

        # classifier is einet, so we need to concatenate the explaining vars
        x = torch.cat([x, exp_vars], dim=1)
        return self.einet(x, marginalized_scopes=self.marginalized_scopes)

    def decode(self, x):
        # decode
        x = self.fc2(x)
        x = x.view(-1, 128, self.image_shape[1], self.image_shape[2])
        x = self.bn3(x)
        x = self.deconv1(x)
        x = self.activation(x)
        x = self.bn2(x)
        x = self.deconv2(x)
        x = self.activation(x)
        x = self.bn1(x)
        x = self.deconv3(x)
        x = torch.sigmoid(x)
        return x

    def start_train(
        self,
        dl_train,
        dl_valid,
        device,
        learning_rate,
        lambda_v,
        warmup_epochs,
        num_epochs,
        lr_schedule_step_size=None,
        lr_schedule_gamma=None,
        early_stop=3,
        checkpoint_dir=None,
        trial=None,
        marginal_divisor=4,
        gamma_v=0.5,
        **kwargs,  # ignore additional arguments
    ):
        # due to its entirely different architecture,
        # the autoencoder needs its own training function

        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        if lr_schedule_step_size is not None and lr_schedule_gamma is not None:
            lr_schedule_resnet = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=lr_schedule_step_size,
                gamma=lr_schedule_gamma,
            )

        val_increase = 0
        lowest_val_loss = torch.inf
        epoch = 0
        t = tqdm(range(warmup_epochs + num_epochs))
        for epoch in t:
            t.set_description(f"Epoch {epoch}")
            loss = 0.0
            for data, target in dl_train:
                print("train")
                optimizer.zero_grad()
                target = target.type(torch.LongTensor)
                data, target = data.to(device), target.to(device)
                output = self(data)  # get LL's
                loss_ce = (
                    lambda_v * torch.nn.CrossEntropyLoss()(output, target)
                    + (1 - lambda_v) * -output.mean()
                )
                sample = self.latent
                if epoch >= warmup_epochs:
                    # stochastic regularization -> marginalize 1/x variables to reconstruct
                    marginalized_scope = torch.randperm(
                        128 + len(self.explaining_vars)
                    )[: ((128 + len(self.explaining_vars)) // marginal_divisor)]
                    # enlargen sample with explaining vars
                    # Note: this currently assumes all explaining vars to be in front
                    sample = torch.cat(
                        [
                            data[:, self.explaining_vars],
                            sample,
                        ],
                        dim=1,
                    )
                    # reconstruct latent space via MPE
                    sample = self.einet.sample(
                        evidence=sample,
                        marginalized_scopes=marginalized_scope,
                        is_differentiable=True,
                        is_mpe=True,
                    )
                    # remove explaining vars from sample
                    sample = sample[:, len(self.explaining_vars) :]
                # decode on (reconstructed) latent space (without explaining vars)
                recon = self.decode(sample)
                # mask out explaining vars
                mask = torch.ones_like(data, dtype=torch.bool)
                mask[:, self.explaining_vars] = False
                data = data[mask].reshape(
                    -1, self.image_shape[0], self.image_shape[1], self.image_shape[2]
                )
                loss_recon = torch.nn.MSELoss()(recon, data)
                loss_v = gamma_v * loss_ce + (1 - gamma_v) * loss_recon
                loss += loss_v.item()
                loss_v.backward()
                optimizer.step()
                break
            if lr_schedule_resnet:
                lr_schedule_resnet.step()

            val_loss = 0.0
            with torch.no_grad():
                for data, target in dl_valid:
                    print("val")
                    optimizer.zero_grad()
                    target = target.type(torch.LongTensor)
                    data, target = data.to(device), target.to(device)
                    output = self(data)
                    loss_ce = (
                        lambda_v * torch.nn.CrossEntropyLoss()(output, target)
                        + (1 - lambda_v) * -output.mean()
                    )
                    sample = self.latent
                    if epoch >= warmup_epochs:
                        # stochastic regularization -> marginalize 1/x variables to reconstruct
                        marginalized_scope = torch.randperm(
                            128 + len(self.explaining_vars)
                        )[: ((128 + len(self.explaining_vars)) // marginal_divisor)]
                        # enlargen sample with explaining vars
                        # Note: this currently assumes all explaining vars to be in front
                        sample = torch.cat(
                            [
                                data[:, self.explaining_vars],
                                sample,
                            ],
                            dim=1,
                        )
                        # reconstruct latent space via MPE
                        sample = self.einet.sample(
                            evidence=sample,
                            marginalized_scopes=marginalized_scope,
                            is_differentiable=True,
                            is_mpe=True,
                        )
                        # remove explaining vars from sample
                        sample = sample[:, len(self.explaining_vars) :]
                    # decode on (reconstructed) latent space
                    recon = self.decode(sample)
                    # mask out explaining vars
                    mask = torch.ones_like(data, dtype=torch.bool)
                    mask[:, self.explaining_vars] = False
                    data = data[mask].reshape(
                        -1,
                        self.image_shape[0],
                        self.image_shape[1],
                        self.image_shape[2],
                    )
                    loss_recon = torch.nn.MSELoss()(recon, data)
                    loss_v = gamma_v * loss_ce + (1 - gamma_v) * loss_recon
                    val_loss += loss_v.item()
                    break
            t.set_postfix(
                dict(
                    train_loss=loss / len(dl_train.dataset),
                    val_loss=val_loss / len(dl_valid.dataset),
                )
            )

            mlflow.log_metric(
                key="train_loss", value=loss / len(dl_train.dataset), step=epoch
            )
            mlflow.log_metric(
                key="val_loss", value=val_loss / len(dl_valid.dataset), step=epoch
            )
            # early stopping via optuna
            if trial:
                trial.report(val_loss, warmup_epochs + epoch)
                if trial.should_prune():
                    mlflow.set_tag("pruned", f"einet: {epoch}")
                    raise optuna.TrialPruned()
            if val_loss <= lowest_val_loss:
                lowest_val_loss = val_loss
                val_increase = 0
                if checkpoint_dir:
                    self.save(checkpoint_dir + "checkpoint.pt")
            else:
                # early stopping when val increases
                val_increase += 1
                if val_increase >= early_stop:
                    print(
                        f"Stopping early, val loss increased for the last {early_stop} epochs."
                    )
                    break
            break

        if checkpoint_dir is not None:
            # load best
            self.load(checkpoint_dir + "checkpoint.pt")
        # log reconstruction of first 2 images
        with torch.no_grad():
            x, y = next(iter(dl_valid))
            x, y = x.to(device), y.to(device)
            output = self(x)
            marginalized_scope = torch.randperm(128 + len(self.explaining_vars))[
                : ((128 + len(self.explaining_vars)) // marginal_divisor)
            ]
            # enlargen sample with explaining vars
            # Note: this currently assumes all explaining vars to be in front
            sample = torch.cat(
                [
                    x[:, self.explaining_vars],
                    sample,
                ],
                dim=1,
            )
            # reconstruct latent space via MPE
            sample = self.einet.sample(
                evidence=sample,
                marginalized_scopes=marginalized_scope,
                is_differentiable=True,
                is_mpe=True,
            )
            # remove explaining vars from sample
            sample = sample[:, len(self.explaining_vars) :]

            recon = self.decode(self.latent)
            recon_mpe = self.decode(sample)

            # mask out explaining vars
            mask = torch.ones_like(x, dtype=torch.bool)
            mask[:, self.explaining_vars] = False
            x = x[mask]
            # reshape to image
            x = x.reshape(
                -1, self.image_shape[0], self.image_shape[1], self.image_shape[2]
            )
            plt.imshow(np.transpose(x[0].cpu().numpy(), (1, 2, 0)))
            mlflow.log_figure(plt.gcf(), "original.png")
            plt.imshow(np.transpose(recon[0].cpu().numpy(), (1, 2, 0)))
            mlflow.log_figure(plt.gcf(), "recon.png")
            plt.imshow(np.transpose(recon_mpe[0].cpu().numpy(), (1, 2, 0)))
            mlflow.log_figure(plt.gcf(), "recon_mpe.png")
            plt.close("all")
        print("Done")
        return lowest_val_loss


from torchvision.models import efficientnet_v2_s
from torch.nn.utils.parametrizations import spectral_norm


class EfficientNetSPN(nn.Module, EinetUtils):
    """
    Spectral normalized EfficientNetV2-S with einet as the output layer.
    """

    def __init__(
        self,
        num_classes,
        image_shape,  # (C, H, W)
        explaining_vars=[],  # indices of variables that should be explained
        spec_norm_bound=0.9,
        einet_depth=3,
        einet_num_sums=20,
        einet_num_leaves=20,
        einet_num_repetitions=1,
        einet_leaf_type="Normal",
        einet_dropout=0.0,
        **kwargs,
    ):
        super(EfficientNetSPN, self).__init__()
        self.num_classes = num_classes
        self.explaining_vars = explaining_vars
        self.spec_norm_bound = spec_norm_bound
        self.einet_depth = einet_depth
        self.einet_num_sums = einet_num_sums
        self.einet_num_leaves = einet_num_leaves
        self.einet_num_repetitions = einet_num_repetitions
        if einet_leaf_type == "Normal":
            self.einet_leaf_type = Normal
        else:
            raise NotImplementedError
        self.einet_dropout = einet_dropout
        self.image_shape = image_shape
        self.marginalized_scopes = None
        self.backbone = self.make_efficientnet()
        self.einet = self.make_einet_output_layer(
            1280 + len(explaining_vars), num_classes
        )

    def make_efficientnet(self):
        def replace_layers_rec(layer):
            """Recursively apply spectral normalization to Conv and Linear layers."""
            if len(list(layer.children())) == 0:
                if isinstance(layer, torch.nn.Conv2d):
                    layer = spectral_norm(layer)
                    # layer = spectral_norm(layer, norm_bound=self.spec_norm_bound)
                elif isinstance(layer, torch.nn.Linear):
                    layer = spectral_norm(layer)
                    # layer = spectral_norm(layer, norm_bound=self.spec_norm_bound)
            else:
                for child in list(layer.children()):
                    replace_layers_rec(child)

        model = efficientnet_v2_s()
        model.features[0][0] = torch.nn.Conv2d(
            self.image_shape[0],
            24,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
        )
        model.classifier = torch.nn.Linear(1280, self.num_classes)
        # apply spectral normalization
        replace_layers_rec(model)
        return model

    def activate_einet(self, deactivate_backbone=True):
        """
        Activates the einet output layer for second stage training and inference.
        """
        self.einet_active = True
        if deactivate_backbone:
            for param in self.parameters():
                param.requires_grad = False
        else:
            for param in self.parameters():
                param.requires_grad = True
        for param in self.einet.parameters():
            param.requires_grad = True

    def deactivate_einet(self):
        """
        Deactivates the einet output layer for first stage training.
        """
        self.einet_active = False
        for param in self.einet.parameters():
            param.requires_grad = False
            pass

    def make_einet_output_layer(self, in_features, out_features):
        """Uses einet as the output layer."""
        cfg = EinetConfig(
            num_features=in_features,
            num_channels=1,
            depth=self.einet_depth,
            num_sums=self.einet_num_sums,
            num_leaves=self.einet_num_leaves,
            num_repetitions=self.einet_num_repetitions,
            num_classes=out_features,
            leaf_type=self.einet_leaf_type,
            layer_type="einsum",
            dropout=self.einet_dropout,
        )
        model = Einet(cfg)
        return model

    def forward_hidden(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        # x is flattened
        # extract explaining vars
        exp_vars = x[:, self.explaining_vars]
        # mask out explaining vars for resnet
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:, self.explaining_vars] = False
        x = x[mask]
        # reshape to image
        x = x.reshape(-1, self.image_shape[0], self.image_shape[1], self.image_shape[2])

        # feed through resnet
        x = self.forward_hidden(x)

        if self.einet_active:
            # classifier is einet, so we need to concatenate the explaining vars
            x = torch.cat([x, exp_vars], dim=1)
            return self.einet(x, marginalized_scopes=self.marginalized_scopes)

        return self.backbone.classifier(x)  # default classifier
