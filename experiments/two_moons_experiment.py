import torch

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets

from torch.utils.data import DataLoader
import os
import mlflow
from plotting_utils import plot_uncertainty_surface


def make_training_data(sample_size=500):
    """Create two moon training dataset."""
    train_examples, train_labels = sklearn.datasets.make_moons(
        n_samples=2 * sample_size, noise=0.1
    )

    # Adjust data position slightly.
    train_examples[train_labels == 0] += [-0.1, 0.2]
    train_examples[train_labels == 1] += [0.1, -0.2]

    return train_examples.astype(np.float32), train_labels.astype(np.int32)


def make_testing_data():
    """Create a mesh grid in 2D space."""
    plt.rcParams["figure.dpi"] = 140
    DEFAULT_X_RANGE = (-3.5, 3.5)
    DEFAULT_Y_RANGE = (-2.5, 2.5)
    DEFAULT_N_GRID = 100
    x_range = DEFAULT_X_RANGE
    y_range = DEFAULT_Y_RANGE
    n_grid = DEFAULT_N_GRID
    # testing data (mesh grid over data space)
    x = np.linspace(x_range[0], x_range[1], n_grid).astype(np.float32)
    y = np.linspace(y_range[0], y_range[1], n_grid).astype(np.float32)
    xv, yv = np.meshgrid(x, y)
    return np.stack([xv.flatten(), yv.flatten()], axis=-1)


def make_ood_data(sample_size=500, means=(2.5, -1.75), vars=(0.01, 0.01)):
    return np.random.multivariate_normal(
        means, cov=np.diag(vars), size=sample_size
    ).astype(np.float32)


def start_two_moons_run(run_name, batch_sizes, model_params, train_params, trial):
    print("starting new two-moons run: ", run_name)
    with mlflow.start_run(run_name=run_name):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mlflow.log_param("device", device)

        ckpt_dir = f"ckpts/two_moons/{run_name}/"
        os.makedirs(ckpt_dir, exist_ok=True)
        mlflow.log_param("ckpt_dir", ckpt_dir)

        # log all params
        mlflow.log_params(batch_sizes)
        mlflow.log_params(model_params)
        mlflow.log_params(train_params)

        # Load the train, test and OOD datasets.
        train_examples, train_labels = make_training_data(sample_size=500)
        test_examples = make_testing_data()
        ood_examples = make_ood_data(sample_size=500)

        pos_examples = train_examples[train_labels == 0]
        neg_examples = train_examples[train_labels == 1]

        # put into data loaders
        train_ds = list(zip(train_examples, train_labels))
        train_dl = DataLoader(
            train_ds[:900],
            batch_size=batch_sizes["resnet"],
            shuffle=True,
            pin_memory=True,
            num_workers=1,
        )
        valid_dl = DataLoader(
            train_ds[900:],
            batch_size=batch_sizes["resnet"],
            shuffle=True,
            pin_memory=True,
            num_workers=1,
        )

        model_name = model_params["model"]
        del model_params["model"]
        if model_name == "DenseResNetSNGP":
            from Models import DenseResNetSNGP

            model = DenseResNetSNGP(**model_params)
        elif model_name == "DenseResNetSPN":
            from Models import DenseResNetSPN

            model = DenseResNetSPN(**model_params)
        elif model_name == "DenseResNetGMM":
            from Models import DenseResNetGMM

            model = DenseResNetGMM(**model_params)
        mlflow.set_tag("model", model.__class__.__name__)
        print(model)
        model.to(device)
        # it is interesting to play with lambda_v, dropout, repetition and depth
        lowest_val_loss = model.start_train(
            train_dl,
            valid_dl,
            device,
            checkpoint_dir=ckpt_dir,
            trial=trial,
            **train_params,
        )

        if "GMM" in model_name:
            model.fit_gmm(train_dl, device)

        # before costly evaluation, make sure that the model is not completely off
        # valid_acc = model.eval_acc(valid_dl, device)
        # mlflow.log_metric("valid_acc", valid_acc)
        # if valid_acc < 0.5:
        #     # let optuna know that this is a bad trial
        #     return lowest_val_loss

        model.activate_uncert_head()
        mlflow.pytorch.log_state_dict(model.state_dict(), "model")

        # evaluate
        model.eval()

        test_dl = DataLoader(
            test_examples,
            batch_size=batch_sizes["resnet"],
            pin_memory=True,
            num_workers=1,
        )

        # Visualize SPN posterior
        posteriors = model.eval_posterior(None, device, test_dl, return_all=True)
        # posteriors = torch.exp(posteriors)
        # use softmax instead of exp, because we want to normalize the posteriors
        posteriors = torch.softmax(posteriors, dim=1)
        # take the probability of class 0 as the uncertainty
        # if p==1 -> no uncertainty, if p==0 -> high uncertainty
        probs_class_0 = posteriors[:, 0].cpu().detach().numpy()
        fig, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(
            train_examples,
            train_labels,
            probs_class_0,
            ax=ax,
            ood_examples=ood_examples,
        )
        plt.colorbar(pcm, ax=ax)
        mlflow.log_figure(fig, "posterior_class_probability.pdf")

        # Visualize SPN predictive entropy
        entropy = -torch.sum(posteriors * torch.log(posteriors), axis=1)
        entropy = entropy.cpu().detach().numpy()
        print("entropy: ", entropy[:5])
        fig, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(
            train_examples, train_labels, entropy, ax=ax, ood_examples=ood_examples
        )
        plt.colorbar(pcm, ax=ax)
        mlflow.log_figure(fig, "posterior_predictive_entropy.pdf")

        # Visualize SPN predictive variance/uncertainty
        # unnecessary according to fabrizio, because we have the entropy
        variance = probs_class_0 * (1 - probs_class_0)
        print("variance: ", variance[:5])
        fig, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(
            train_examples, train_labels, variance, ax=ax, ood_examples=ood_examples
        )
        plt.colorbar(pcm, ax=ax)
        mlflow.log_figure(fig, "posterior_predictive_variance.pdf")

        lls = model.eval_ll_marg(None, device, test_dl, return_all=True)
        nll = -(lls.cpu().detach().numpy())  # negative log likelihood
        fig, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(
            train_examples, train_labels, nll, ax=ax, ood_examples=ood_examples
        )
        plt.colorbar(pcm, ax=ax)
        mlflow.log_figure(fig, "nll.pdf")

        fig, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(
            train_examples,
            train_labels,
            nll,
            ax=ax,
            plot_train=False,
            ood_examples=ood_examples,
        )
        plt.colorbar(pcm, ax=ax)
        mlflow.log_figure(fig, "nll_no_train.pdf")

        dempster_shafer = (
            model.eval_dempster_shafer(None, device, test_dl, return_all=True)
            .cpu()
            .detach()
            .numpy()
        )
        fig, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(
            train_examples,
            train_labels,
            dempster_shafer,
            ax=ax,
            ood_examples=ood_examples,
        )
        plt.colorbar(pcm, ax=ax)
        mlflow.log_figure(fig, "dempster_shafer.pdf")

        # maximum predictive probability as in Figure 3, appendix C in https://arxiv.org/pdf/2006.10108.pdf
        max_pred_prob = np.max(posteriors.cpu().detach().numpy(), axis=1)
        print("max posterior: ", max_pred_prob[:5])
        uncertainty = 1 - 2 * np.abs(max_pred_prob - 0.5)
        print("uncert max posterior: ", uncertainty[:5])
        fig, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(
            train_examples, train_labels, uncertainty, ax=ax, ood_examples=ood_examples
        )
        plt.colorbar(pcm, ax=ax)
        mlflow.log_figure(fig, "max_pred_prob.pdf")

        # Test epistemic as in DDU p(z)
        # p_z = torch.exp(lls)
        p_z = torch.exp(lls - torch.logsumexp(lls, dim=0))

        epistemic = p_z.cpu().detach().numpy()
        print("density p(z): ", epistemic[:5])
        print("min density p(z): ", np.min(epistemic))
        print("max density p(z): ", np.max(epistemic))

        fig, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(
            train_examples, train_labels, epistemic, ax=ax, ood_examples=ood_examples
        )
        plt.colorbar(pcm, ax=ax)
        mlflow.log_figure(fig, "epistemic.pdf")
        fig, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(
            train_examples,
            train_labels,
            epistemic,
            ax=ax,
            plot_train=False,
            ood_examples=ood_examples,
        )
        plt.colorbar(pcm, ax=ax)
        mlflow.log_figure(fig, "epistemic_notrain.pdf")

        # Test aleatoric as in DDU entropy of softmax
        logits = model.backbone_logits(test_dl, device, return_all=True)
        probs = torch.softmax(logits, dim=1)
        aleatoric = -torch.sum(probs * torch.log(probs), axis=1).cpu().detach().numpy()
        print(aleatoric)
        fig, ax = plt.subplots(figsize=(7, 5.5))
        pcm = plot_uncertainty_surface(
            train_examples, train_labels, aleatoric, ax=ax, ood_examples=ood_examples
        )
        plt.colorbar(pcm, ax=ax)
        mlflow.log_figure(fig, "aleatoric.pdf")

        plt.close()
        return lowest_val_loss
