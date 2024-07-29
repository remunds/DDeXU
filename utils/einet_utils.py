import os
import torch

from tqdm import tqdm
import numpy as np
import mlflow
import optuna
import matplotlib.pyplot as plt
from sklearn import metrics


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
                self.load(pretrained_path, map_location=device)
            else:
                self.load(pretrained_path, map_location=device)
            # self.deactivate_uncert_head()
            # val_acc = self.eval_acc(dl_valid, device)
            # print(f"pretrained backbone validation accuracy: {val_acc}")
            # mlflow.log_metric(key="pretrained_backbone_val_acc", value=val_acc)
            # self.activate_uncert_head(deactivate_backbone)
            # val_acc = self.eval_acc(dl_valid, device)
            # print(f"pretrained model validation accuracy: {val_acc}")
            # mlflow.log_metric(key="pretrained_val_acc", value=val_acc)
        if num_epochs == 0 and warmup_epochs == 0:
            return 0.0

        self.train()
        self.deactivate_uncert_head()
        print("lr: ", learning_rate_warmup)
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
                    target = target.type(torch.LongTensor)
                    data, target = data.to(device), target.to(device)
                    output = self(data)
                    loss_v = torch.nn.CrossEntropyLoss()(output, target)
                    val_loss += loss_v.item()
            t.set_postfix(
                dict(
                    train_loss=loss / (len(dl_train.dataset)),
                    val_loss=val_loss / (len(dl_valid.dataset)),
                )
            )
            mlflow.log_metric(
                key="warmup_train_loss", value=loss / len(dl_train.dataset), step=epoch
            )
            mlflow.log_metric(
                key="warmup_val_loss",
                value=val_loss / len(dl_valid.dataset),
                step=epoch,
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

        # compute norm and std
        self.compute_normalization_values(dl_train, device)

        if num_epochs == 0:
            return lowest_val_loss

        # train einet (and optionally resnet jointly)
        self.activate_uncert_head(deactivate_backbone)
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
        divisor = self.num_hidden + len(self.explaining_vars)
        t = tqdm(range(num_epochs))
        for epoch in t:
            t.set_description(f"Epoch {warmup_epochs_performed + epoch}")
            loss = 0.0
            ce_loss = 0.0
            nll_loss = 0.0
            for data, target in dl_train:
                optimizer.zero_grad()
                target = target.type(torch.LongTensor)
                data, target = data.to(device), target.to(device)
                output = self(data)
                # output: logS(x|y_i) Shape: (N, C)

                # generative training uses NLL: -1/n sum_n logS(x) (across all samples)
                # logS(x) = logsumexp_y logS(x|y) + logP(y)
                # logP(y) = log(1/C) (uniform prior)
                # logS(x) = logsumexp_y logS(x|y) + log(1/C)
                # output_mean = 1/N sum_n 1/C sum_c logS(x|y_c) = 1/N sum_n logS(x)
                # so that works!

                # Also: we can use the CE of S(x|y) here, because with the assumption
                # of uniform prior, CE of S(x|y) is equivalent to S(y|x) (see equation 4 RAT paper)

                # The division factor |X| is equation 5 of RAT.
                # This is the dimension of input to SPN, so hidden + explaining vars

                ce_loss_v = lambda_v * torch.nn.CrossEntropyLoss()(output, target)
                ce_loss += ce_loss_v.item()
                nll_loss_v = (1 - lambda_v) * -(output.mean() / divisor)
                nll_loss += nll_loss_v.item()

                loss_v = ce_loss_v + nll_loss_v

                loss += loss_v.item()
                loss_v.backward()
                optimizer.step()
            if lr_schedule_einet is not None:
                lr_schedule_einet.step()
            val_loss = 0.0
            val_ce_loss = 0.0
            val_nll_loss = 0.0
            with torch.no_grad():
                for data, target in dl_valid:
                    target = target.type(torch.LongTensor)
                    data, target = data.to(device), target.to(device)
                    output = self(data)
                    ce_loss_v = lambda_v * torch.nn.CrossEntropyLoss()(output, target)
                    val_ce_loss += ce_loss_v.item()
                    nll_loss_v = (1 - lambda_v) * -(output.mean() / divisor)
                    val_nll_loss += nll_loss_v.item()
                    loss_v = ce_loss_v + nll_loss_v
                    val_loss += loss_v.item()
                t.set_postfix(
                    dict(
                        train_loss=loss / len(dl_train.dataset),
                        val_loss=val_loss / len(dl_valid.dataset),
                    )
                )
                mlflow.log_metric(
                    key="ce_loss_train",
                    value=ce_loss / len(dl_train.dataset),
                    step=warmup_epochs_performed + epoch,
                )
                mlflow.log_metric(
                    key="nll_loss_train",
                    value=nll_loss / len(dl_train.dataset),
                    step=warmup_epochs_performed + epoch,
                )
                mlflow.log_metric(
                    key="train_loss",
                    value=loss / len(dl_train.dataset),
                    step=warmup_epochs_performed + epoch,
                )
                mlflow.log_metric(
                    key="ce_loss_val",
                    value=val_ce_loss / len(dl_valid.dataset),
                    step=warmup_epochs_performed + epoch,
                )
                mlflow.log_metric(
                    key="nll_loss_val",
                    value=val_nll_loss / len(dl_valid.dataset),
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

    def load(self, path, backbone_only=False, map_location=None):
        state_dict = torch.load(path, map_location=map_location)
        if backbone_only:
            print("loading backbone only")
            state_dict = {k: v for k, v in state_dict.items() if "einet" not in k}
        self.load_state_dict(state_dict, strict=False)
        # self.load_state_dict(state_dict)

    def get_embeddings(self, dl, device):
        self.eval()
        embedding_size = self.num_hidden + len(self.explaining_vars)
        index = 0
        embeddings = torch.zeros(len(dl.dataset), embedding_size).to(device)
        with torch.no_grad():
            for data in dl:
                if type(data) is tuple or type(data) is list:
                    data = data[0]
                data = data.to(device)
                output = self(data)  # required for hidden
                expl_vars = data[:, self.explaining_vars]
                curr_embeddings = torch.concat([expl_vars, self.hidden], dim=1)
                embeddings[index : index + len(output)] = curr_embeddings
                index += len(output)
        return embeddings

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
        """
        Returns logP(x|y_i)
        - log likelihood of all classes.
        Output tensor has shape (N, C) or (C)
        """
        self.eval()
        index = 0
        lls = torch.zeros(len(dl.dataset), self.num_classes).to(device)
        with torch.no_grad():
            for data in dl:
                if type(data) is tuple or type(data) is list:
                    data = data[0]
                data = data.to(device)
                ll = self(data)
                lls[index : index + len(ll)] = ll
                index += len(ll)
        if return_all:
            return lls  # (N, C)
        return torch.mean(lls, dim=0)  # (C)

    def eval_ll_marg(self, log_p_x_g_y, device, dl=None, return_all=False):
        """
        Returns logP(x)
        - marginal log likelihood of the input data.
        P(x) = sum_y P(x,y) = sum_y P(x|y)P(y)
        or in log:
        log P(x) = logsumexp_y log P(x,y) = logsumexp_y log P(x|y) + log P(y)
        Output tensor has shape (N) or is scalar

        """
        self.eval()
        assert log_p_x_g_y is not None or dl is not None
        if log_p_x_g_y is None:
            # use dl to compute log_p_x_g_y
            log_p_x_g_y = self.eval_ll(dl, device, return_all=True)
        log_y = torch.log(torch.tensor(1.0 / self.num_classes)).to(device)
        log_p_x_y = log_p_x_g_y + log_y
        # marginalize over y
        log_p_x = torch.logsumexp(log_p_x_y, dim=1)
        if return_all:
            return log_p_x  # (N)
        return torch.mean(log_p_x).item()  # scalar

    def backbone_logits(self, dl, device, return_all=False):
        self.deactivate_uncert_head()
        self.eval()
        index = 0
        logits = torch.zeros(len(dl.dataset), self.num_classes).to(device)
        with torch.no_grad():
            for data in dl:
                if type(data) is tuple or type(data) is list:
                    data = data[0]
                data = data.to(device)
                output = self(data)
                logits[index : index + len(output)] = output
                index += len(output)
        self.activate_uncert_head()
        if return_all:
            return logits  # (N, C)
        return torch.mean(logits, dim=0)  # (C)

    # def eval_posterior(self, dl, device, return_all=False):
    def eval_posterior(self, log_p_x_g_y, device, dl=None, return_all=False):
        self.eval()
        assert log_p_x_g_y is not None or dl is not None
        if log_p_x_g_y is None:
            log_p_x_g_y = self.eval_ll(dl, device, return_all=True)
        from simple_einet.einet import posterior

        posteriors = posterior(log_p_x_g_y, self.num_classes)
        if return_all:
            return posteriors  # (N, C)
        return torch.mean(posteriors, dim=0)  # (C)

    def eval_entropy(self, log_p_x_g_y, device, dl=None, return_all=False):
        assert log_p_x_g_y is not None or dl is not None
        if log_p_x_g_y is None:
            # use dl to compute log_p_x_g_y
            log_p_x_g_y = self.eval_ll(dl, device, return_all=True)
        posteriors_log = self.eval_posterior(log_p_x_g_y, device, dl, return_all=True)
        # check if contains nan
        # use softmax to convert logs to probs
        posteriors = torch.softmax(posteriors_log, dim=1)
        entropy = -torch.sum(posteriors * posteriors_log, dim=1)
        if return_all:
            return entropy  # (N)
        return torch.mean(entropy).item()  # scalar

    def eval_highest_class_prob(self, log_p_x_g_y, device, dl=None, return_all=False):
        assert log_p_x_g_y is not None or dl is not None
        if log_p_x_g_y is None:
            # use dl to compute log_p_x_g_y
            log_p_x_g_y = self.eval_ll(dl, device, return_all=True)
        posteriors_log = self.eval_posterior(log_p_x_g_y, device, dl, return_all=True)
        # use softmax to convert logs to probs
        posteriors = torch.softmax(posteriors_log, dim=1)
        highest_prob = torch.max(posteriors, dim=1)[0]
        if return_all:
            return highest_prob  # (N)
        return torch.mean(highest_prob).item()  # scalar

    def eval_correct_class_prob(self, log_p_x_g_y, device, dl, return_all=False):
        assert dl is not None
        if log_p_x_g_y is None:
            # use dl to compute log_p_x_g_y
            log_p_x_g_y = self.eval_ll(dl, device, return_all=True)
        posteriors_log = self.eval_posterior(log_p_x_g_y, device, dl, return_all=True)
        # use softmax to convert logs to probs
        posteriors = torch.softmax(posteriors_log, dim=1)
        # get correct class
        labels = (
            torch.cat([labels for _, labels in dl], dim=0).to(device).to(torch.long)
        )
        correct_class_prob = posteriors[
            torch.arange(len(labels), device=device), labels
        ]
        if return_all:
            return correct_class_prob  # (N)
        return torch.mean(correct_class_prob).item()  # scalar

    def eval_dempster_shafer(self, log_p_x_g_y, device, dl=None, return_all=False):
        """
        SNGP: better for distance aware models, where
        magnitude of logits reflects distance from observed data manifold
        https://arxiv.org/pdf/2006.10108.pdf
        """
        self.eval()
        assert log_p_x_g_y is not None or dl is not None
        if log_p_x_g_y is None:
            # use dl to compute log_p_x_g_y
            log_p_x_g_y = self.eval_ll(dl, device, return_all=True)
        posteriors_log = self.eval_posterior(log_p_x_g_y, device, dl, return_all=True)
        uncertainty = self.num_classes / (
            self.num_classes + torch.sum(torch.exp(posteriors_log), dim=1)
        )
        if return_all:
            return uncertainty  # (N)
        return torch.mean(uncertainty).item()  # scalar

    def eval_calibration(
        self, log_p_x_g_y, device, name, dl, n_bins=20, method="posterior"
    ):
        """Computes the expected calibration error and plots the calibration curve."""
        self.eval()
        assert dl is not None
        if log_p_x_g_y is None:
            # use dl to compute log_p_x_g_y
            log_p_x_g_y = self.eval_ll(dl, device, return_all=True)

        # get posteriors p(y_i | x) via bayes rule
        posteriors_log = self.eval_posterior(log_p_x_g_y, device, dl, return_all=True)
        # (N, C)
        # use softmax to convert logs to probs
        probs = torch.softmax(posteriors_log, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
        if method == "entropy":
            name += "_entropy"
            # replace confidences with entropies
            entropies = self.eval_entropy(log_p_x_g_y, device, dl, return_all=True)
            confidences = entropies * -1
            # normalize to [0, 1]
            confidences = (confidences - torch.min(confidences)) / (
                torch.max(confidences) - torch.min(confidences)
            )

        if method == "nll":
            name += "_nll"
            nll = -log_p_x_g_y.mean(dim=1)
            # normalize to [0, 1]
            nll = (nll - torch.min(nll)) / (torch.max(nll) - torch.min(nll))
            confidences = nll

        print(confidences[:5])
        from plotting_utils import histogram_plot, calibration_plot

        # make a histogram of the confidences
        histogram_plot(confidences.cpu().numpy(), n_bins, name)

        # Note: this assumes that the dl does not shuffle the data
        labels = torch.cat([labels for _, labels in dl], dim=0).to(device)

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

        def compute_ece(conf, acc):
            eces = []
            for i in range(len(conf)):
                if (
                    np.isnan(acc[i])
                    or np.isnan(conf[i])
                    or acc[i] is None
                    or conf[i] is None
                ):
                    continue
                eces.append(abs(acc[i] - conf[i]))
            ece = sum(eces) / len(eces)
            return ece

        # NLL = -logP(x|y)
        # nll = -self.eval_ll(dl, device, return_all=False)
        # nll = - sum log von richtigem entry
        mean_nll = -torch.mean(
            torch.log(
                self.eval_correct_class_prob(log_p_x_g_y, device, dl, return_all=True)
            )
        )
        mlflow.log_metric(key=f"mean_nll_{name}", value=mean_nll)

        # equal frequency binning
        conf_freq, acc_freq = equal_frequency_binning(
            confidences, predictions, labels, num_bins=n_bins
        )
        ece = compute_ece(conf_freq, acc_freq)
        mlflow.log_metric(key=f"ece_eq_freq_{name}", value=ece)
        calibration_plot(conf_freq, acc_freq, ece, mean_nll, f"eq_freq_{name}")
        conf_w, acc_w = equal_width_binning(
            confidences, predictions, labels, num_bins=n_bins
        )
        ece = compute_ece(conf_w, acc_w)
        mlflow.log_metric(key=f"ece_eq_width_{name}", value=ece)
        calibration_plot(conf_w, acc_w, ece, mean_nll, f"eq_width_{name}")
        return (conf_freq, acc_freq), (conf_w, acc_w)

    # taken from https://github.com/omegafragger/DDU/blob/main/metrics/ood_metrics.py
    def eval_ood(self, uncert_id, uncert_ood, device, confidence=False):
        uncertainties = uncert_id
        ood_uncertainties = uncert_ood

        # In-distribution
        bin_labels = torch.zeros(uncertainties.shape[0]).to(device)
        in_scores = uncertainties

        # OOD
        bin_labels = torch.cat(
            (bin_labels, torch.ones(ood_uncertainties.shape[0]).to(device))
        )

        if confidence:
            bin_labels = 1 - bin_labels
        ood_scores = ood_uncertainties  # entropy(ood_logits)
        scores = torch.cat((in_scores, ood_scores))

        fpr, tpr, thresholds = metrics.roc_curve(
            bin_labels.cpu().numpy(), scores.cpu().numpy()
        )
        precision, recall, prc_thresholds = metrics.precision_recall_curve(
            bin_labels.cpu().numpy(), scores.cpu().numpy()
        )
        auroc = metrics.roc_auc_score(bin_labels.cpu().numpy(), scores.cpu().numpy())
        auprc = metrics.average_precision_score(
            bin_labels.cpu().numpy(), scores.cpu().numpy()
        )

        return (fpr, tpr, thresholds), (precision, recall, prc_thresholds), auroc, auprc

    def explain_ll(self, dl, device, return_all=False):
        """
        Check each explaining variable individually.
        Returns the difference in data log likelihood between the default model and the marginalized model.
        Use, when the explaining variables are given in the data.
        Captures only aleatoric uncertainty.
        """
        ll_default = self.eval_ll_marg(None, device, dl, return_all)
        explanations = []
        for i in self.explaining_vars:
            self.marginalized_scopes = [i]
            ll_marg = self.eval_ll_marg(None, device, dl, return_all)
            explanations.append((ll_marg - ll_default))
        self.marginalized_scopes = None
        if return_all:
            return torch.stack(explanations, dim=1)
        return explanations

    def explain_posterior(self, dl, device, return_all=False):
        """
        Check each explaining variable individually.
        Returns the difference in posterior entropy between the default model and the marginalized model.
        Use, when the explaining variables are given in the data.
        Captures both aleatoric and epistemic uncertainty.
        """
        # Intuition: the entropy should be higher when the explaining variable is observed
        # than in the counterfactual case, where it is marginalized.
        # What if the variable wasn't there? -> lower entropy because more certain
        entropy_default = self.eval_entropy(None, device, dl, return_all)
        explanations = []
        for i in self.explaining_vars:
            self.marginalized_scopes = [i]
            entropy_marg = self.eval_entropy(None, device, dl, return_all)
            explanations.append((entropy_default - entropy_marg))
        self.marginalized_scopes = None
        if return_all:
            return torch.stack(explanations, dim=1)
        return explanations

    def explain_mpe(self, dl, device, return_all=False):
        """
        Explain the explaining variables by computing their most probable explanation (MPE).
        Use, when the explaining variables are not given in the data.
        """
        assert (
            self.mean is not None and self.std is not None
        ), "call compute_normalization_values first"
        expl_var_mpes = []
        for data, _ in dl:
            data = data.to(device)
            # extract explaining vars
            exp_vars = data[:, self.explaining_vars]
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
            # Normalize hidden
            hidden = torch.cat([exp_vars, hidden], dim=1)
            hidden = self.normalize(hidden)
            hidden[:, self.explaining_vars] = 0.0
            mpe: torch.Tensor = self.einet.mpe(
                evidence=hidden, marginalized_scopes=self.explaining_vars
            )
            mpe = self.denormalize(mpe)
            expl_var_mpes.append(mpe[:, self.explaining_vars])
        if return_all:
            return expl_var_mpes
        return torch.cat(expl_var_mpes, dim=0).mean(dim=0)

    def compute_normalization_values(self, dl, device):
        """
        Computes mean and std per dim
        Resulting shapes will be (D,)
        """
        print("computing normalization values")
        self.eval()
        self.deactivate_uncert_head()
        embeddings = self.get_embeddings(dl, device)
        self.mean = embeddings.mean(axis=0)
        self.std = embeddings.std(axis=0)
        mean_expl = self.mean[self.explaining_vars]
        std_expl = self.std[self.explaining_vars]
        print(f"mean: {mean_expl}, std: {std_expl}")

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x is raw form of explaining-vars + embedding
        Returns them normalized in [0,1].
        """
        assert (
            self.mean is not None and self.std is not None
        ), "call compute_normalization_values first"
        self.std[self.std == 0] = 0.000001
        x_norm = (x - self.mean) / self.std
        return x_norm

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x is normalized form of explaining-vars + embedding
        Returns denormalized form of these.
        """
        assert (
            self.mean is not None and self.std is not None
        ), "call compute_normalization_values first"
        if type(x) is np.ndarray:
            x_denorm = x * self.std.cpu().numpy() + self.mean.cpu().numpy()
        else:
            x_denorm = x * self.std + self.mean
        return x_denorm

    def dequantize(self, explanations: torch.Tensor) -> torch.Tensor:
        """
        Dequantizes the explaining variables from discrete space to continuous space.
        This is common in normalizing flows and aids the training.

        Flow++ paper: Adding uniform noise to the discrete data over the width of each
        discrete bin: if each of the D components of the discrete data
        x takes on values in {0, 1, 2, . . . , 255}, then the dequantized
        data is given by y = x+u, where u is drawn uniformly from [0, 1)^D

        Theis et. al 2015: maximizing dequantized log-likelihood is lower bound to max. log-likelihood of data
        """
        # add uniform noise [0, 1)
        expl_dequant = explanations + torch.rand(explanations.shape).to(
            explanations.device
        )
        return expl_dequant

    def embedding_histogram(self, dl, device):
        self.eval()
        embeddings = self.get_embeddings(dl, device)
        exp_vars = embeddings[:, : len(self.explaining_vars)]
        exp_vars = self.dequantize(exp_vars)
        embeddings[:, : len(self.explaining_vars)] = exp_vars
        # shape: (Dataset-size, num_hidden + len(explaining_vars))
        for i in range(embeddings.shape[1]):
            plt.hist(embeddings[:, i].cpu().numpy(), bins=50)
            plt.title(f"Embedding {i}")
            mlflow.log_figure(plt.gcf(), f"embedding_{i}.png")
            plt.clf()
