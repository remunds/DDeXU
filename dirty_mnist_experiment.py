import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import ddu_dirty_mnist
import mlflow


def get_datasets():
    data_dir = "/data_docker/datasets/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mnist_transform = transforms.Compose(
        [
            # transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.reshape(-1, 28 * 28).squeeze()),
        ]
    )
    fashion_mnist_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.reshape(-1, 28 * 28).squeeze()),
        ]
    )

    # load dirty mnist
    train_ds = ddu_dirty_mnist.DirtyMNIST(
        data_dir + "dirty_mnist",
        train=True,
        transform=mnist_transform,
        download=True,
        normalize=False,
        device=device,
    )
    test_ds = ddu_dirty_mnist.DirtyMNIST(
        data_dir + "dirty_mnist",
        train=False,
        transform=mnist_transform,
        download=True,
        normalize=False,
        device=device,
    )

    ambiguous_ds_test = ddu_dirty_mnist.AmbiguousMNIST(
        data_dir + "dirty_mnist",
        train=False,
        transform=mnist_transform,
        download=True,
        normalize=False,
        device=device,
    )

    mnist_ds_test = ddu_dirty_mnist.FastMNIST(
        data_dir + "dirty_mnist",
        train=False,
        transform=mnist_transform,
        download=True,
        normalize=False,
        device=device,
    )

    ood_ds = datasets.FashionMNIST(
        data_dir + "fashionmnist",
        train=False,
        download=True,
        transform=fashion_mnist_transform,
    )

    train_ds, valid_ds = torch.utils.data.random_split(
        train_ds, [0.8, 0.2], generator=torch.Generator().manual_seed(0)
    )

    return train_ds, valid_ds, test_ds, ambiguous_ds_test, mnist_ds_test, ood_ds


def start_dirty_mnist_run(run_name, batch_sizes, model_params, train_params, trial):
    with mlflow.start_run(run_name=run_name) as run:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlflow.log_param("device", device)

        ckpt_dir = f"/data_docker/ckpts/dirty_mnist/{run_name}/"
        os.makedirs(ckpt_dir, exist_ok=True)
        mlflow.log_param("ckpt_dir", ckpt_dir)

        # log all params
        mlflow.log_params(batch_sizes)
        mlflow.log_params(model_params)
        mlflow.log_params(train_params)

        # load data
        (
            train_ds,
            valid_ds,
            test_ds,
            ambiguous_ds_test,
            mnist_ds_test,
            ood_ds,
        ) = get_datasets()

        # create dataloaders
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_sizes["resnet"],
            shuffle=True,
            pin_memory=False,
            num_workers=0,
        )
        valid_dl = DataLoader(
            valid_ds,
            batch_size=batch_sizes["resnet"],
            shuffle=True,
            pin_memory=False,
            num_workers=0,
        )
        test_dl = DataLoader(
            test_ds,
            batch_size=batch_sizes["resnet"],
            shuffle=True,
            pin_memory=False,
            num_workers=0,
        )

        # create model
        model_name = model_params["model"]
        del model_params["model"]
        if model_name == "ConvResNetSPN":
            from ResNetSPN import ConvResNetSPN, ResidualBlockSN, BottleNeckSN

            if model_params["block"] == "basic":
                block = ResidualBlockSN
            elif model_params["block"] == "bottleneck":
                block = BottleNeckSN
            else:
                raise NotImplementedError

            del model_params["block"]
            layers = model_params["layers"]
            del model_params["layers"]
            del model_params["spectral_normalization"]
            del model_params["mod"]

            model = ConvResNetSPN(
                block,
                layers,
                explaining_vars=[],  # for calibration test, we don't need explaining vars
                **model_params,
            )
        elif model_name == "ConvResNetDDU":
            from ResNetSPN import ConvResnetDDU
            from net.resnet import BasicBlock, Bottleneck

            if model_params["block"] == "basic":
                block = BasicBlock
            elif model_params["block"] == "bottleneck":
                block = Bottleneck
            else:
                raise NotImplementedError

            del model_params["block"]
            layers = model_params["layers"]
            del model_params["layers"]
            del model_params["spec_norm_bound"]
            model = ConvResnetDDU(
                block,
                layers,
                explaining_vars=[],  # for calibration test, we don't need explaining vars
                **model_params,
            )
        elif model_name == "AutoEncoderSPN":
            from ResNetSPN import AutoEncoderSPN

            model = AutoEncoderSPN(
                explaining_vars=[],  # for calibration test, we don't need explaining vars
                **model_params,
            )
        elif model_name == "EfficientNetSPN":
            from ResNetSPN import EfficientNetSPN

            model = EfficientNetSPN(
                explaining_vars=[],  # for calibration test, we don't need explaining vars
                **model_params,
            )
        else:
            raise NotImplementedError
        mlflow.set_tag("model", model.__class__.__name__)
        model = model.to(device)

        print("training model")
        # train model
        lowest_val_loss = model.start_train(
            train_dl,
            valid_dl,
            device,
            checkpoint_dir=ckpt_dir,
            trial=trial,
            **train_params,
        )
        mlflow.pytorch.log_state_dict(model.state_dict(), "model")

        # evaluate
        model.eval()
        model.einet_active = False
        train_acc = model.eval_acc(train_dl, device)
        mlflow.log_metric("train accuracy resnet", train_acc)
        test_acc = model.eval_acc(test_dl, device)
        mlflow.log_metric("test accuracy resnet", test_acc)

        model.einet_active = True
        train_acc = model.eval_acc(train_dl, device)
        mlflow.log_metric("train accuracy", train_acc)
        test_acc = model.eval_acc(test_dl, device)
        mlflow.log_metric("test accuracy", test_acc)

        train_ll = model.eval_ll(train_dl, device)
        mlflow.log_metric("train ll", train_ll)
        test_ll = model.eval_ll(test_dl, device)
        mlflow.log_metric("test ll", test_ll)

        # create dataloaders
        mnist_dl = DataLoader(
            mnist_ds_test,
            batch_size=batch_sizes["resnet"],
            shuffle=False,
            pin_memory=False,
            num_workers=0,
        )
        ambiguous_dl = DataLoader(
            ambiguous_ds_test,
            batch_size=batch_sizes["resnet"],
            shuffle=False,
            pin_memory=False,
            num_workers=0,
        )
        ood_dl = DataLoader(
            ood_ds,
            batch_size=batch_sizes["resnet"],
            shuffle=False,
            pin_memory=False,
            num_workers=0,
        )

        # plot as in DDU paper
        # three datasets: mnist, ambiguous mnist, ood
        # x-axis: log-likelihood, y-axis: fraction of data

        # get log-likelihoods for all datasets
        with torch.no_grad():
            lls_mnist = model.eval_ll(mnist_dl, device, return_all=True)
            mlflow.log_metric("mnist ll", torch.mean(lls_mnist).item())
            pred_entropy_mnist = model.eval_entropy(mnist_dl, device, return_all=True)
            mlflow.log_metric("mnist entropy", torch.mean(pred_entropy_mnist).item())
            highest_class_prob_mnist = model.eval_highest_class_prob(
                mnist_dl, device, return_all=True
            )
            mlflow.log_metric(
                "mnist highest class prob", torch.mean(highest_class_prob_mnist).item()
            )
            correct_class_prob_mnist = model.eval_correct_class_prob(
                mnist_dl, device, return_all=True
            )
            mlflow.log_metric(
                "mnist correct class prob", torch.mean(correct_class_prob_mnist).item()
            )
            lls_unmarg = model.eval_ll_unmarginalized(mnist_dl, device, return_all=True)
            lls_lsexp = torch.logsumexp(lls_unmarg, axis=1)
            p_z_y0 = torch.exp(lls_unmarg[:, 0] - lls_lsexp)
            p_z_y1 = torch.exp(lls_unmarg[:, 1] - lls_lsexp)
            epistemic_mnist = p_z_y0 * 0.5 + p_z_y1 * 0.5  # density of embedding z
            mlflow.log_metric("mnist epistemic", torch.mean(epistemic_mnist).item())

            backbone_logits = model.backbone_logits(mnist_dl, device, return_all=True)
            probs = torch.softmax(backbone_logits, dim=1)
            aleatoric_mnist = -torch.sum(probs * torch.log(probs), dim=1)
            mlflow.log_metric("mnist aleatoric", torch.mean(aleatoric_mnist).item())

            lls_amb = model.eval_ll(ambiguous_dl, device, return_all=True)
            mlflow.log_metric("ambiguous ll", torch.mean(lls_amb).item())
            pred_entropy_amb = model.eval_entropy(ambiguous_dl, device, return_all=True)
            mlflow.log_metric("ambiguous entropy", torch.mean(pred_entropy_amb).item())
            highest_class_prob_amb = model.eval_highest_class_prob(
                ambiguous_dl, device, return_all=True
            )
            mlflow.log_metric(
                "ambiguous highest class prob",
                torch.mean(highest_class_prob_amb).item(),
            )
            correct_class_prob_amb = model.eval_correct_class_prob(
                ambiguous_dl, device, return_all=True
            )
            mlflow.log_metric(
                "ambiguous correct class prob",
                torch.mean(correct_class_prob_amb).item(),
            )
            lls_unmarg = model.eval_ll_unmarginalized(
                ambiguous_dl, device, return_all=True
            )
            lls_lsexp = torch.logsumexp(lls_unmarg, axis=1)
            p_z_y0 = torch.exp(lls_unmarg[:, 0] - lls_lsexp)
            p_z_y1 = torch.exp(lls_unmarg[:, 1] - lls_lsexp)
            epistemic_amb = p_z_y0 * 0.5 + p_z_y1 * 0.5  # density of embedding z
            mlflow.log_metric("ambiguous epistemic", torch.mean(epistemic_amb).item())

            backbone_logits = model.backbone_logits(
                ambiguous_dl, device, return_all=True
            )
            probs = torch.softmax(backbone_logits, dim=1)
            aleatoric_amb = -torch.sum(probs * torch.log(probs), dim=1)
            mlflow.log_metric("ambiguous aleatoric", torch.mean(aleatoric_amb).item())

            lls_ood = model.eval_ll(ood_dl, device, return_all=True)
            mlflow.log_metric("ood ll", torch.mean(lls_ood).item())
            pred_entropy_ood = model.eval_entropy(ood_dl, device, return_all=True)
            mlflow.log_metric("ood entropy", torch.mean(pred_entropy_ood).item())
            highest_class_prob_ood = model.eval_highest_class_prob(
                ood_dl, device, return_all=True
            )
            mlflow.log_metric(
                "ood highest class prob", torch.mean(highest_class_prob_ood).item()
            )
            correct_class_prob_ood = model.eval_correct_class_prob(
                ood_dl, device, return_all=True
            )
            mlflow.log_metric(
                "ood correct class prob", torch.mean(correct_class_prob_ood).item()
            )
            lls_unmarg = model.eval_ll_unmarginalized(ood_dl, device, return_all=True)
            lls_lsexp = torch.logsumexp(lls_unmarg, axis=1)
            p_z_y0 = torch.exp(lls_unmarg[:, 0] - lls_lsexp)
            p_z_y1 = torch.exp(lls_unmarg[:, 1] - lls_lsexp)
            epistemic_ood = p_z_y0 * 0.5 + p_z_y1 * 0.5
            mlflow.log_metric("ood epistemic", torch.mean(epistemic_ood).item())

            backbone_logits = model.backbone_logits(ood_dl, device, return_all=True)
            probs = torch.softmax(backbone_logits, dim=1)
            aleatoric_ood = -torch.sum(probs * torch.log(probs), dim=1)
            mlflow.log_metric("ood aleatoric", torch.mean(aleatoric_ood).item())

        # plot
        def hist_plot(
            data_mnist, data_amb, data_ood, xlabel, ylabel, filename, bins=30
        ):
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set_style("whitegrid")
            sns.set_context("paper", font_scale=1.5)
            # plot mnist
            fig, ax = plt.subplots()
            sns.histplot(
                data_mnist.cpu().numpy(),
                stat="probability",
                label="MNIST",
                ax=ax,
                bins=bins,
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            # plot ambiguous mnist
            sns.histplot(
                data_amb.cpu().numpy(),
                stat="probability",
                label="Ambiguous MNIST",
                ax=ax,
                bins=bins,
            )

            # plot ood
            sns.histplot(
                data_ood.cpu().numpy(),
                stat="probability",
                label="Fashion MNIST",
                bins=bins,
                ax=ax,
            )

            fig.legend()
            fig.tight_layout()
            mlflow.log_figure(fig, filename)
            plt.close()

        # Likelihood plot
        hist_plot(
            lls_mnist,
            lls_amb,
            lls_ood,
            "Log-likelihood",
            "Fraction of data",
            "mnist_amb_ood_ll.png",
        )

        # Predictive entropy plot
        hist_plot(
            pred_entropy_mnist,
            pred_entropy_amb,
            pred_entropy_ood,
            "Entropy",
            "Fraction of data",
            "mnist_amb_ood_entropy.png",
        )

        # Highest class probability plot
        hist_plot(
            highest_class_prob_mnist,
            highest_class_prob_amb,
            highest_class_prob_ood,
            "Highest class probability",
            "Fraction of data",
            "mnist_amb_ood_highest_class_prob.png",
        )

        # Correct class probability plot
        hist_plot(
            correct_class_prob_mnist,
            correct_class_prob_amb,
            correct_class_prob_ood,
            "Correct class probability",
            "Fraction of data",
            "mnist_amb_ood_correct_class_prob.png",
        )

        # Epistemic plot
        hist_plot(
            epistemic_mnist,
            epistemic_amb,
            epistemic_ood,
            "Epistemic uncertainty",
            "Fraction of data",
            "mnist_amb_ood_epistemic.png",
        )

        # Aleatoric plot
        hist_plot(
            aleatoric_mnist,
            aleatoric_amb,
            aleatoric_ood,
            "Aleatoric uncertainty",
            "Fraction of data",
            "mnist_amb_ood_aleatoric.png",
        )

        return lowest_val_loss
