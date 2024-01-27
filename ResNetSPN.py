import os
from typing import Any
import torch

from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

# from net.resnet import ResNet, BasicBlock, Bottleneck
import torch.nn as nn
from spectral_normalization import spectral_norm
from simple_einet.einet import EinetConfig, Einet

from simple_einet.layers.distributions.normal import Normal, RatNormal
from simple_einet.layers.distributions.categorical import Categorical
from simple_einet.layers.distributions.multidistribution import MultiDistributionLayer
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
        early_stop=5,
        checkpoint_dir=None,
        trial=None,
        pretrained_path=None,
    ):
        if pretrained_path is not None:
            print(f"loading pretrained model from {pretrained_path}")
            if num_epochs == 0 and warmup_epochs == 0:
                # load complete model and return
                self.load(pretrained_path)
            else:
                self.load(pretrained_path, backbone_only=True)
            self.deactivate_einet()
            val_acc = self.eval_acc(dl_valid, device)
            print(f"pretrained backbone validation accuracy: {val_acc}")
            mlflow.log_metric(key="pretrained_backbone_val_acc", value=val_acc)
            self.activate_einet(deactivate_backbone)
            val_acc = self.eval_acc(dl_valid, device)
            print(f"pretrained model validation accuracy: {val_acc}")
            mlflow.log_metric(key="pretrained_val_acc", value=val_acc)
            if num_epochs == 0 and warmup_epochs == 0:
                return 0.0

        self.train()
        self.deactivate_einet()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate_warmup)
        lr_schedule_backbone = None
        if (
            lr_schedule_warmup_step_size is not None
            and lr_schedule_warmup_gamma is not None
        ):
            lr_schedule_backbone = torch.optim.lr_scheduler.StepLR(
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
            if lr_schedule_backbone is not None:
                lr_schedule_backbone.step()

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
            if trial is not None:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    mlflow.set_tag("pruned", f"backbone: {epoch}")
                    raise optuna.TrialPruned()
            if val_loss <= lowest_val_loss:
                lowest_val_loss = val_loss
                val_increase = 0
                if checkpoint_dir is not None:
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
            print("loading best backbone checkpoint")
            self.load(checkpoint_dir + "checkpoint.pt")

        # train einet (and optionally resnet jointly)
        self.activate_einet(deactivate_backbone)
        if deactivate_backbone:
            optimizer = torch.optim.Adam(self.einet.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        lr_schedule_einet = None
        if lr_schedule_step_size is not None and lr_schedule_gamma is not None:
            lr_schedule_einet = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=lr_schedule_step_size,
                gamma=lr_schedule_gamma,
            )
        lowest_val_loss = torch.inf if num_epochs > 0 else lowest_val_loss
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
            if lr_schedule_einet is not None:
                lr_schedule_einet.step()
            # print(5000 * mpe_loss)
            # print(loss_v - (5000 * mpe_loss))
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
                if trial is not None:
                    trial.report(val_loss, warmup_epochs + epoch)
                    if trial.should_prune():
                        mlflow.set_tag("pruned", f"einet: {epoch}")
                        raise optuna.TrialPruned()
                if val_loss <= lowest_val_loss:
                    lowest_val_loss = val_loss
                    val_increase = 0
                    if checkpoint_dir is not None:
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
        state_dict = torch.load(path)
        if backbone_only:
            state_dict = {k: v for k, v in state_dict.items() if "einet" not in k}
        self.load_state_dict(state_dict, strict=False)

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
            for data in dl:
                if type(data) is tuple or type(data) is list:
                    data = data[0]
                data = data.to(device)
                ll = self(data)
                lls[index : index + len(ll)] = ll.mean(dim=1)
                index += len(ll)
        if return_all:
            return lls
        return torch.mean(lls).item()

    def eval_posterior(self, dl, device, return_all=False):
        self.eval()
        index = 0
        posteriors = torch.zeros(len(dl.dataset), self.num_classes)
        with torch.no_grad():
            for data in dl:
                if type(data) is tuple or type(data) is list:
                    data = data[0]
                data = data.to(device)
                lls = self(data)  # this sets the einet_input
                posterior = self.einet.posterior(self.einet_input)
                posteriors[index : index + len(posterior)] = posterior
                index += len(posterior)
        if return_all:
            return posteriors
        return torch.mean(posteriors).item()

    def eval_entropy(self, dl, device, return_all=False):
        posteriors_log = self.eval_posterior(dl, device, return_all=True)
        # convert log probs to probs
        posteriors = torch.exp(posteriors_log)
        entropy = -torch.sum(posteriors * posteriors_log, dim=1)
        if return_all:
            return entropy
        return torch.mean(entropy).item()

    def eval_highest_class_prob(self, dl, device, return_all=False):
        posteriors_log = self.eval_posterior(dl, device, return_all=True)
        # convert log probs to probs
        posteriors = torch.exp(posteriors_log)
        highest_prob = torch.max(posteriors, dim=1)[0]
        if return_all:
            return highest_prob
        return torch.mean(highest_prob).item()

    def eval_correct_class_prob(self, dl, device, return_all=False):
        posteriors_log = self.eval_posterior(dl, device, return_all=True)
        # convert log probs to probs
        posteriors = torch.exp(posteriors_log)
        # get correct class
        labels = torch.cat([labels for _, labels in dl], dim=0)
        correct_class_prob = posteriors[torch.arange(len(labels)), labels]
        if return_all:
            return correct_class_prob
        return torch.mean(correct_class_prob).item()

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

    # def eval_pred_entropy(self, dl, device, return_all=False):
    #     self.eval()
    #     index = 0
    #     pred_entropies = torch.zeros(len(dl.dataset))
    #     with torch.no_grad():
    #         for data, _ in dl:
    #             data = data.to(device)
    #             pred_logit = self(data)
    #             pred = torch.softmax(pred_logit, dim=1)
    #             pred_entropy = -torch.sum(pred * torch.log(pred), dim=1)
    #             pred_entropies[index : index + len(pred_entropy)] = pred_entropy
    #             index += len(pred_entropy)
    #     if return_all:
    #         return pred_entropies
    #     return torch.mean(pred_entropies).item()

    def eval_dempster_shafer(self, dl, device, return_all=False):
        """
        SNGP: better for distance aware models, where
        magnitude of logits reflects distance from observed data manifold
        https://arxiv.org/pdf/2006.10108.pdf
        """
        num_classes = self.einet.config.num_classes
        self.eval()
        index = 0
        uncertainties = torch.zeros(len(dl.dataset))
        with torch.no_grad():
            for data in dl:
                if type(data) is tuple or type(data) is list:
                    data = data[0]
                data = data.to(device)
                lls = self(data)
                posterior = self.einet.posterior(self.einet_input)
                uncertainty = num_classes / (
                    num_classes + torch.sum(torch.exp(posterior), dim=1)
                )
                uncertainties[index : index + len(uncertainty)] = uncertainty
                index += len(uncertainty)
        if return_all:
            return uncertainties
        return torch.mean(uncertainties).item()

    def explain_ll(self, dl, device, return_all=False):
        """
        Check each explaining variable individually.
        Returns the difference in log likelihood between the default model and the marginalized model.
        If largely positive, the variable explains the result.
        """
        ll_default = self.eval_ll(dl, device, return_all)
        explanations = []
        for i in self.explaining_vars:
            self.marginalized_scopes = [i]
            ll_marg = self.eval_ll(dl, device, return_all)
            explanations.append((ll_marg - ll_default))
        self.marginalized_scopes = None
        if return_all:
            return torch.stack(explanations, dim=1)
        return explanations

    def explain_mpe(self, dl, device, return_all=False):
        """
        Aktuelle idee (benoetigt gegebene expl. var. daten):
        Vergleiche MPE der expl. vars mit evidence
        mit den tatsaechlich gegebenen Daten
        Aussage von starken Unterschieden: "Anders als durch bisherige Distr. erwartet"
        Weitere Idee (benoetigt keine gegebenen expl. var. daten):
        Vergleiche MPE der expl. vars mit evidence
        mit dem MPE der expl. vars ohne evidence
        -> wenn hoeher: starker unterschied zu Distribution
        """
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
                # ConvResNets
                # reshape to image
                data = data.reshape(
                    -1, self.image_shape[0], self.image_shape[1], self.image_shape[2]
                )
            else:
                # DenseResNetSPN
                data = data.reshape(-1, self.input_dim)

            # extract most probable explanation of current input
            hidden = self.forward_hidden(data)
            hidden = torch.cat([exp_vars, hidden], dim=1)
            mpe = self.einet.mpe(
                evidence=hidden, marginalized_scopes=self.explaining_vars
            )
            expl_var_mpes.append(mpe[:, self.explaining_vars])
        expl_var_vals = torch.cat(expl_var_vals, dim=0)
        expl_var_mpes = torch.cat(expl_var_mpes, dim=0)
        if return_all:
            return expl_var_vals, expl_var_mpes
        return torch.abs(expl_var_mpes - expl_var_vals).mean(dim=0).cpu().numpy()

    def eval_calibration(self, dl, device, name, n_bins=10):
        """Computes the expected calibration error and plots the calibration curve."""
        self.eval()

        # get posteriors p(y_i | x) via bayes rule
        posteriors = self.eval_posterior(dl, device, return_all=True)

        # get confidences and predictions via max/argmax
        confidences, predictions = torch.max(posteriors, dim=1)
        # convert from logit to probability
        confidences = torch.exp(confidences)
        # make a histogram of the confidences
        plt.hist(confidences.cpu().numpy(), bins=n_bins)
        mlflow.log_figure(plt.gcf(), f"ll_hist_{name}.png")
        plt.clf()

        # predictions = torch.argmax(posteriors, dim=1)

        # this assumes that the dl does not shuffle the data
        labels = torch.cat([labels for _, labels in dl], dim=0)

        def equal_frequency_binning(confidences, predictions, labels, num_bins):
            # Sort confidences and predictions
            sorted_indices = torch.argsort(confidences)
            sorted_confidences = confidences[sorted_indices]
            sorted_predictions = predictions[sorted_indices]
            sorted_labels = labels[sorted_indices]

            # Split data into equal frequency bins
            binned_conf = torch.chunk(sorted_confidences, num_bins)
            binned_pred = torch.chunk(sorted_predictions, num_bins)
            binned_labels = torch.chunk(sorted_labels, num_bins)

            bin_confidences = []
            bin_accuracies = []

            # Calculate observed accuracy for each bin
            for bin_c, bin_p, bin_l in zip(binned_conf, binned_pred, binned_labels):
                bin_confidences.append(torch.mean(bin_c).item())
                bin_accuracies.append((bin_p == bin_l).to(float).mean().item())

            return bin_confidences, bin_accuracies

        def equal_width_binning(confidences, predictions, labels, num_bins):
            # Sort confidences and predictions
            sorted_indices = torch.argsort(confidences)
            sorted_confidences = confidences[sorted_indices]
            sorted_predictions = predictions[sorted_indices]
            sorted_labels = labels[sorted_indices]

            # Split data into bins in range [0, 1] with equal width
            bin_size = 1 / num_bins
            bin_confidences = []
            bin_accuracies = []

            for i in range(num_bins):
                # Get indices of samples in current bin
                bin_indices = torch.where(
                    (sorted_confidences >= i * bin_size)
                    & (sorted_confidences < (i + 1) * bin_size)
                )[0]
                # Calculate mean confidence and accuracy for current bin
                bin_confidences.append(
                    torch.mean(sorted_confidences[bin_indices]).item()
                )
                bin_accuracies.append(
                    (sorted_predictions[bin_indices] == sorted_labels[bin_indices])
                    .to(float)
                    .mean()
                    .item()
                )

            return bin_confidences, bin_accuracies

        def plot_calibration_curve(conf, acc, ece, nll, title):
            # Plot the calibration curve
            plt.plot(conf, acc, marker="o")
            plt.plot(
                [0, 1], [0, 1], linestyle="--", color="gray"
            )  # Diagonal line for reference
            plt.xlabel("Mean Confidence")
            plt.ylabel("Observed Accuracy")
            plt.title(
                "Calibration Plot, ECE: {ece:.3f}, NLL: {nll:.3f}".format(
                    ece=ece, nll=nll
                )
            )
            mlflow.log_figure(plt.gcf(), f"calibration_curve_{title}.png")
            plt.clf()

        def compute_ece(conf, acc):
            ece = 0.0
            for i in range(len(conf)):
                if acc[i] == torch.nan or conf[i] == torch.nan:
                    continue
                ece += abs(acc[i] - conf[i])
            ece = ece / len(conf)
            return ece

        # the negative ll's is simply the mean output of the einet across all samples
        # output is likelihood: P(x | y_i) for each class y_i
        # taking the mean over the classes gives the marginal likelihood P(x)
        # do that for each sample and take the mean over all samples
        nll = -self.eval_ll(dl, device, return_all=False)

        # equal frequency binning
        conf, acc = equal_frequency_binning(
            confidences, predictions, labels, num_bins=n_bins
        )
        ece = compute_ece(conf, acc)
        plot_calibration_curve(conf, acc, ece, nll, f"equal_frequency_{name}")
        conf, acc = equal_width_binning(
            confidences, predictions, labels, num_bins=n_bins
        )
        ece = compute_ece(conf, acc)
        plot_calibration_curve(conf, acc, ece, nll, f"equal_width_{name}")


class DenseResnet(nn.Module):
    """
    A simple fully connected ResNet.
    """

    def __init__(
        self,
        input_dim,
        num_classes,
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
        self.num_classes = num_classes

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
        return nn.Linear(self.num_hidden, self.num_classes, **self.classifier_kwargs)


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
        leaf_type = RatNormal
        leaf_kwargs = {
            "min_sigma": 0.00001,
            "max_sigma": 10.0,
        }
        cfg = EinetConfig(
            num_features=self.num_hidden + len(self.explaining_vars),
            num_channels=1,
            depth=self.einet_depth,
            num_sums=self.einet_num_sums,
            num_leaves=self.einet_num_leaves,
            num_repetitions=self.einet_num_repetitions,
            num_classes=self.num_classes,
            leaf_type=leaf_type,
            leaf_kwargs=leaf_kwargs,
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
            hidden = torch.cat([exp_vars, hidden], dim=1)
            self.einet_input = hidden
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
        if len(self.explaining_vars) > 0:
            leaf_type = MultiDistributionLayer
            scopes_a = torch.arange(0, len(self.explaining_vars))
            scopes_b = torch.arange(
                len(self.explaining_vars), 1280 + len(self.explaining_vars)
            )
            leaf_kwargs = {
                "scopes_to_dist": [
                    (scopes_a, RatNormal, {"min_mean": 0.0, "max_mean": 6.0}),
                    # (scopes_a, Categorical, {"num_bins": 5}),
                    (
                        scopes_b,
                        RatNormal,
                        {"min_sigma": 0.00001, "max_sigma": 10.0},
                    ),  # Tuple of (scopes, class, kwargs)
                ]
            }
        else:
            leaf_type = RatNormal
            leaf_kwargs = {"min_sigma": 0.00001, "max_sigma": 10.0}
        cfg = EinetConfig(
            num_features=in_features,
            num_channels=1,
            depth=self.einet_depth,
            num_sums=self.einet_num_sums,
            num_leaves=self.einet_num_leaves,
            num_repetitions=self.einet_num_repetitions,
            num_classes=out_features,
            # leaf_type=self.einet_leaf_type,
            leaf_type=leaf_type,
            leaf_kwargs=leaf_kwargs,
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
            x = torch.cat([exp_vars, x], dim=1)
            self.einet_input = x
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
        if len(self.explaining_vars) > 0:
            leaf_type = MultiDistributionLayer
            scopes_a = torch.arange(0, len(self.explaining_vars))
            scopes_b = torch.arange(
                len(self.explaining_vars), 1280 + len(self.explaining_vars)
            )
            leaf_kwargs = {
                "scopes_to_dist": [
                    (scopes_a, RatNormal, {"min_mean": 0.0, "max_mean": 6.0}),
                    # (scopes_a, Categorical, {"num_bins": 5}),
                    (
                        scopes_b,
                        RatNormal,
                        {"min_sigma": 0.00001, "max_sigma": 10.0},
                    ),  # Tuple of (scopes, class, kwargs)
                ]
            }
        else:
            leaf_type = RatNormal
            leaf_kwargs = {"min_sigma": 0.00001, "max_sigma": 10.0}
        cfg = EinetConfig(
            num_features=in_features,
            num_channels=1,
            depth=self.einet_depth,
            num_sums=self.einet_num_sums,
            num_leaves=self.einet_num_leaves,
            num_repetitions=self.einet_num_repetitions,
            num_classes=out_features,
            # leaf_type=self.einet_leaf_type,
            leaf_type=leaf_type,
            leaf_kwargs=leaf_kwargs,
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
            x = torch.cat([exp_vars, x], dim=1)
            self.einet_input = x
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
        x = torch.cat([exp_vars, x], dim=1)
        self.einet_input = x
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
        lr_schedule_backbone = None
        if lr_schedule_step_size is not None and lr_schedule_gamma is not None:
            lr_schedule_backbone = torch.optim.lr_scheduler.StepLR(
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
            if lr_schedule_backbone is not None:
                lr_schedule_backbone.step()

            val_loss = 0.0
            with torch.no_grad():
                for data, target in dl_valid:
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
            if trial is not None:
                trial.report(val_loss, warmup_epochs + epoch)
                if trial.should_prune():
                    mlflow.set_tag("pruned", f"einet: {epoch}")
                    raise optuna.TrialPruned()
            if val_loss <= lowest_val_loss:
                lowest_val_loss = val_loss
                val_increase = 0
                if checkpoint_dir is not None:
                    self.save(checkpoint_dir + "checkpoint.pt")
            else:
                # early stopping when val increases
                val_increase += 1
                if val_increase >= early_stop:
                    print(
                        f"Stopping early, val loss increased for the last {early_stop} epochs."
                    )
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
            # RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 512 but got size 224 for tensor number 1 in the list.
            sample = self.latent
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
        return lowest_val_loss


from torchvision.models import efficientnet_v2_s
from torch.nn.utils.parametrizations import spectral_norm as spectral_norm_torch


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
            # self.einet_leaf_type = Normal
            self.einet_leaf_type = RatNormal
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
                    layer = spectral_norm_torch(layer)
                    # layer = spectral_norm(layer, norm_bound=self.spec_norm_bound)
                elif isinstance(layer, torch.nn.Linear):
                    layer = spectral_norm_torch(layer)
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
        if len(self.explaining_vars) > 0:
            leaf_type = MultiDistributionLayer
            scopes_a = torch.arange(0, len(self.explaining_vars))
            scopes_b = torch.arange(
                len(self.explaining_vars), 1280 + len(self.explaining_vars)
            )
            leaf_kwargs = {
                "scopes_to_dist": [
                    (scopes_a, RatNormal, {"min_mean": 0.0, "max_mean": 6.0}),
                    # (scopes_a, Categorical, {"num_bins": 5}),
                    (
                        scopes_b,
                        RatNormal,
                        {"min_sigma": 0.00001, "max_sigma": 10.0},
                    ),  # Tuple of (scopes, class, kwargs)
                ]
            }
        else:
            leaf_type = RatNormal
            leaf_kwargs = {"min_sigma": 0.00001, "max_sigma": 10.0}

        cfg = EinetConfig(
            num_features=in_features,
            num_channels=1,
            depth=self.einet_depth,
            num_sums=self.einet_num_sums,
            num_leaves=self.einet_num_leaves,
            num_repetitions=self.einet_num_repetitions,
            num_classes=out_features,
            # leaf_type=self.einet_leaf_type,
            leaf_type=leaf_type,
            leaf_kwargs=leaf_kwargs,
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
        self.hidden = x

        if self.einet_active:
            # classifier is einet, so we need to concatenate the explaining vars
            x = torch.cat([exp_vars, x], dim=1)
            self.einet_input = x
            return self.einet(x, marginalized_scopes=self.marginalized_scopes)

        return self.backbone.classifier(x)  # default classifier
