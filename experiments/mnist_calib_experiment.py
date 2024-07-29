import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import os
import mlflow
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def get_severity(test_ds, i):
    severity_levels = 5

    # if we have 5k images, 1k per severity level
    index_1 = len(test_ds) // severity_levels
    index_2 = index_1 * 2
    index_3 = index_1 * 3
    index_4 = index_1 * 4
    return (
        1
        if i < index_1
        else 2 if i < index_2 else 3 if i < index_3 else 4 if i < index_4 else 5
    )


def get_datasets():
    data_dir = "/data_docker/datasets/"

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

    print("manipulating images")

    # 1: 20, 2: 40, 3: 60, 4: 80, 5: 100 degrees
    rotations = torch.zeros((len(test_ds),))
    for i, r in enumerate(rotations):
        severity = get_severity(test_ds, i)
        rotations[i] = severity * 20

    # 1: 2, 2: 4, 3: 9, 4: 16, 5: 25 pixels
    cutoffs = torch.zeros((len(test_ds),))
    for i, r in enumerate(cutoffs):
        severity = get_severity(test_ds, i)
        cutoffs[i] = 2 if severity == 1 else severity**2

    # 0: 10, 1: 20, 2: 30, 3: 40, 4: 50 noise
    noises = torch.zeros((len(test_ds),))
    for i, r in enumerate(noises):
        severity = get_severity(test_ds, i)
        noises[i] = severity * 10

    test_ds_rot = torch.zeros((len(test_ds), 28 * 28))
    test_ds_cutoff = torch.zeros((len(test_ds), 28 * 28))
    test_ds_noise = torch.zeros((len(test_ds), 28 * 28))
    for i, img in enumerate(test_ds.data):
        image = img.reshape(28, 28, 1).clone()

        this_noise = torch.randn((28, 28, 1)) * noises[i]
        img_noise = torch.clamp(image + this_noise, 0, 255).to(dtype=torch.uint8)
        img_noise = transforms.ToTensor()(img_noise.numpy())
        test_ds_noise[i] = transforms.Normalize((0.1307,), (0.3081,))(
            img_noise
        ).flatten()

        image_cutoff = image.clone()
        # cutoff rows
        image_cutoff[: int(cutoffs[i]), ...] = 0
        image_cutoff = transforms.ToTensor()(image_cutoff.numpy())
        test_ds_cutoff[i] = transforms.Normalize((0.1307,), (0.3081,))(
            image_cutoff
        ).flatten()

        image_rot = transforms.ToTensor()(image.numpy())
        image_rot = transforms.functional.rotate(
            img=image_rot, angle=int(rotations[i])  # , fill=-mean / std
        )
        test_ds_rot[i] = transforms.Normalize((0.1307,), (0.3081,))(image_rot).flatten()

    print("done manipulating images")

    train_ds, _ = torch.utils.data.random_split(
        train_ds, [0.8, 0.2], generator=torch.Generator().manual_seed(0)
    )
    _, valid_ds = torch.utils.data.random_split(
        valid_ds, [0.8, 0.2], generator=torch.Generator().manual_seed(0)
    )

    return (
        train_ds,
        valid_ds,
        test_ds,
        ood_ds,
        test_ds_rot,
        test_ds_cutoff,
        test_ds_noise,
    )


def start_mnist_calib_run(run_name, batch_sizes, model_params, train_params, trial):
    with mlflow.start_run(run_name=run_name):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlflow.log_param("device", device)

        ckpt_dir = f"/data_docker/ckpts/mnist_calib/{run_name}/"
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
            ood_ds,
            test_ds_rot,
            test_ds_cutoff,
            test_ds_noise,
        ) = get_datasets()

        # create dataloaders
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_sizes["resnet"],
            shuffle=True,
            pin_memory=True,
            num_workers=2,
        )
        valid_dl = DataLoader(
            valid_ds,
            batch_size=batch_sizes["resnet"],
            shuffle=True,
            pin_memory=True,
            num_workers=2,
        )
        test_dl = DataLoader(test_ds, batch_size=batch_sizes["resnet"], shuffle=True)

        # create model
        model_name = model_params["model"]
        del model_params["model"]
        if model_name == "ConvResNetSPN":
            from Models import ConvResNetSPN, ResidualBlockSN, BottleNeckSN

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
            from Models import ConvResnetDDU
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
            from Models import AutoEncoderSPN

            model = AutoEncoderSPN(
                explaining_vars=[],  # for calibration test, we don't need explaining vars
                **model_params,
            )
        elif model_name == "EfficientNetSPN":
            from Models import EfficientNetSPN

            model = EfficientNetSPN(
                explaining_vars=[],  # for calibration test, we don't need explaining vars
                **model_params,
            )
        elif model_name == "EfficientNetGMM":
            from Models import EfficientNetGMM

            model = EfficientNetGMM(
                explaining_vars=[],  # for calibration test, we don't need explaining vars
                **model_params,
            )
        elif model_name == "EfficientNetSNGP":
            from Models import EfficientNetSNGP

            train_num_data = len(train_ds) + len(valid_ds)
            model = EfficientNetSNGP(
                explaining_vars=[],  # for calibration test, we don't need explaining vars
                train_num_data=train_num_data,
                **model_params,
            )
        elif model_name == "ConvResnetDDUGMM":
            from Models import ConvResnetDDUGMM
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
            model = ConvResnetDDUGMM(
                block,
                layers,
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
        # before costly evaluation, make sure that the model is not completely off
        valid_acc = model.eval_acc(valid_dl, device)
        mlflow.log_metric("valid_acc", valid_acc)

        if valid_acc < 0.5:
            # let optuna know that this is a bad trial
            return lowest_val_loss
        if "GMM" in model_name:
            model.fit_gmm(train_dl, device)
        else:
            mlflow.pytorch.log_state_dict(model.state_dict(), "model")

        # evaluate
        model.eval()
        model.deactivate_uncert_head()
        train_acc = model.eval_acc(train_dl, device)
        mlflow.log_metric("train accuracy resnet", train_acc)
        test_acc = model.eval_acc(test_dl, device)
        mlflow.log_metric("test accuracy resnet", test_acc)

        model.activate_uncert_head()
        train_acc = model.eval_acc(train_dl, device)
        mlflow.log_metric("train accuracy", train_acc)
        test_acc = model.eval_acc(test_dl, device)
        mlflow.log_metric("test accuracy", test_acc)

        train_ll_marg = model.eval_ll_marg(None, device, train_dl)
        mlflow.log_metric("train ll marg", train_ll_marg)
        test_ll = model.eval_ll(None, device, return_all=True)
        test_ll_marg = model.eval_ll_marg(test_ll, device)
        mlflow.log_metric("test ll marg", test_ll_marg)
        test_entropy = model.eval_entropy(test_ll, device, return_all=True)
        mlflow.log_metric("test entropy", torch.mean(test_entropy))

        ood_ds_flat = ood_ds.data.reshape(-1, 28 * 28).to(dtype=torch.float32)
        ood_ds_flat = TensorDataset(ood_ds_flat, ood_ds.targets)
        ood_dl = DataLoader(ood_ds_flat, batch_size=batch_sizes["resnet"], shuffle=True)
        ood_ll = model.eval_ll(ood_dl, device, return_all=True)
        ood_ll_marg = model.eval_ll_marg(ood_ll, device)
        mlflow.log_metric("ood ll marg", ood_ll_marg)

        print("evaluating calibration")
        model.eval_calibration(test_ll, device, "test", test_dl)

        # OOD eval
        ood_entropy = model.eval_entropy(ood_ll, device, return_all=True)
        mlflow.log_metric("fashion_entropy", torch.mean(ood_entropy))

        (_, _, _), (_, _, _), auroc, auprc = model.eval_ood(
            test_entropy, ood_entropy, device
        )

        del (
            ood_ds_flat,
            ood_dl,
            train_dl,
            valid_dl,
            test_dl,
            train_ds,
            valid_ds,
        )

        print("eval calibration plots")
        # calibration plots
        test_ds_rot = TensorDataset(test_ds_rot, test_ds.targets)
        test_dl_rot = DataLoader(
            test_ds_rot,
            batch_size=batch_sizes["resnet"],
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )
        model.eval_calibration(None, device, "rotation", dl=test_dl_rot)
        del test_dl_rot

        test_ds_cutoff = TensorDataset(test_ds_cutoff, test_ds.targets)
        test_dl_cutoff = DataLoader(
            test_ds_cutoff,
            batch_size=batch_sizes["resnet"],
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )
        model.eval_calibration(None, device, "cutoff", dl=test_dl_cutoff)
        del test_dl_cutoff

        test_ds_noise = TensorDataset(test_ds_noise, test_ds.targets)
        test_dl_noise = DataLoader(
            test_ds_noise,
            batch_size=batch_sizes["resnet"],
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )

        model.eval_calibration(None, device, "noise", dl=test_dl_noise)
        del test_dl_noise
        print("done creating calibration plots")

        severity_levels = 5

        # if we have 5k images, 1k per severity level
        index_1 = len(test_ds) // severity_levels
        index_2 = index_1 * 2
        index_3 = index_1 * 3
        index_4 = index_1 * 4
        index_5 = index_1 * 5

        # plots by severity
        eval_dict = {}
        severity_indices = [index_1, index_2, index_3, index_4, index_5]
        prev_s = 0
        for s in tqdm(severity_indices):
            test_ds_rot_s = TensorDataset(
                test_ds_rot[prev_s:s][0], test_ds.targets[prev_s:s]
            )
            test_dl_rot = DataLoader(
                test_ds_rot_s,
                batch_size=batch_sizes["resnet"],
                shuffle=False,
                num_workers=1,
                pin_memory=True,
            )

            severity = get_severity(test_ds, s - 1)
            model.deactivate_uncert_head()
            backbone_acc = model.eval_acc(test_dl_rot, device)
            model.activate_uncert_head()
            einet_acc = model.eval_acc(test_dl_rot, device)
            ll = model.eval_ll(test_dl_rot, device, return_all=True)
            ll_marg = model.eval_ll_marg(ll, device)
            entropy = model.eval_entropy(ll, device)
            highest_class_prob = model.eval_highest_class_prob(ll, device)
            # correct_class_prob = model.eval_correct_class_prob(test_dl_rot, device)
            # dempster_shafer = model.eval_dempster_shafer(test_dl_rot, device)

            if "rotation" not in eval_dict:
                eval_dict["rotation"] = {}
            eval_dict["rotation"][severity] = {
                "backbone_acc": backbone_acc,
                "einet_acc": einet_acc,
                "ll_marg": ll_marg,
                "entropy": entropy,
                "highest_class_prob": highest_class_prob,
                # "correct_class_prob": correct_class_prob,
                # "dempster_shafer": dempster_shafer,
            }

            test_ds_cutoff_s = TensorDataset(
                test_ds_cutoff[prev_s:s][0], test_ds.targets[prev_s:s]
            )
            test_dl_cutoff = DataLoader(
                test_ds_cutoff_s,
                batch_size=batch_sizes["resnet"],
                shuffle=False,
                num_workers=1,
                pin_memory=True,
            )
            severity = get_severity(test_ds, s - 1)
            model.deactivate_uncert_head()
            backbone_acc = model.eval_acc(test_dl_cutoff, device)
            model.activate_uncert_head()
            einet_acc = model.eval_acc(test_dl_cutoff, device)
            ll = model.eval_ll(test_dl_cutoff, device, return_all=True)
            ll_marg = model.eval_ll_marg(ll, device)
            entropy = model.eval_entropy(ll, device)
            highest_class_prob = model.eval_highest_class_prob(ll, device)
            # correct_class_prob = model.eval_correct_class_prob(test_dl_cutoff, device)
            # dempster_shafer = model.eval_dempster_shafer(test_dl_cutoff, device)

            if "cutoff" not in eval_dict:
                eval_dict["cutoff"] = {}
            eval_dict["cutoff"][severity] = {
                "backbone_acc": backbone_acc,
                "einet_acc": einet_acc,
                "ll_marg": ll_marg,
                "entropy": entropy,
                "highest_class_prob": highest_class_prob,
                # "correct_class_prob": correct_class_prob,
                # "dempster_shafer": dempster_shafer,
            }

            test_ds_noise_s = TensorDataset(
                test_ds_noise[prev_s:s][0], test_ds.targets[prev_s:s]
            )
            # test_ds_noise_s = test_ds_noise[prev_s:s]
            test_dl_noise = DataLoader(
                test_ds_noise_s,
                batch_size=batch_sizes["resnet"],
                shuffle=False,
                num_workers=1,
                pin_memory=True,
            )
            severity = get_severity(test_ds, s - 1)
            model.deactivate_uncert_head()
            einet_acc = model.eval_acc(test_dl_noise, device)
            model.activate_uncert_head()
            backbone_acc = model.eval_acc(test_dl_noise, device)
            ll = model.eval_ll(test_dl_noise, device, return_all=True)
            ll_marg = model.eval_ll_marg(ll, device)
            entropy = model.eval_entropy(ll, device)
            highest_class_prob = model.eval_highest_class_prob(ll, device)
            # correct_class_prob = model.eval_correct_class_prob(test_dl_noise, device)
            # dempster_shafer = model.eval_dempster_shafer(test_dl_noise, device)

            if "noise" not in eval_dict:
                eval_dict["noise"] = {}
            eval_dict["noise"][severity] = {
                "backbone_acc": backbone_acc,
                "einet_acc": einet_acc,
                "ll_marg": ll_marg,
                "entropy": entropy,
                "highest_class_prob": highest_class_prob,
                # "correct_class_prob": correct_class_prob,
                # "dempster_shafer": dempster_shafer,
            }

            prev_s = s

        mlflow.log_dict(eval_dict, "eval_dict")

        backbone_acc = np.mean(
            [
                eval_dict[m][severity]["backbone_acc"]
                for m in eval_dict
                for severity in eval_dict[m]
            ]
        )
        einet_acc = np.mean(
            [
                eval_dict[m][severity]["einet_acc"]
                for m in eval_dict
                for severity in eval_dict[m]
            ]
        )
        ll_marg = np.mean(
            [
                eval_dict[m][severity]["ll_marg"]
                for m in eval_dict
                for severity in eval_dict[m]
            ]
        )
        entropy = np.mean(
            [
                eval_dict[m][severity]["entropy"]
                for m in eval_dict
                for severity in eval_dict[m]
            ]
        )
        highest_class_prob = np.mean(
            [
                eval_dict[m][severity]["highest_class_prob"]
                for m in eval_dict
                for severity in eval_dict[m]
            ]
        )
        # correct_class_prob = np.mean(
        #     [
        #         eval_dict[m][severity]["correct_class_prob"]
        #         for m in eval_dict
        #         for severity in eval_dict[m]
        #     ]
        # )
        # dempster_shafer = np.mean(
        #     [
        #         eval_dict[m][severity]["dempster_shafer"]
        #         for m in eval_dict
        #         for severity in eval_dict[m]
        #     ]
        # )

        mlflow.log_metric("manip_backbone_acc", backbone_acc)
        mlflow.log_metric("manip_einet_acc", einet_acc)
        mlflow.log_metric("manip_ll_marg", ll_marg)
        mlflow.log_metric("manip_entropy", entropy)
        mlflow.log_metric("manip_highest_class_prob", highest_class_prob)
        # mlflow.log_metric("manip_correct_class_prob", correct_class_prob)
        # mlflow.log_metric("manip_dempster_shafer", dempster_shafer)

        # create plot for each corruption
        # x axis: severity
        # y axis: metrics

        for m in ["rotation", "cutoff", "noise"]:
            einet_acc = [
                eval_dict[m][severity]["einet_acc"]
                for severity in sorted(eval_dict[m].keys())
            ]
            backbone_acc = [
                eval_dict[m][severity]["backbone_acc"]
                for severity in sorted(eval_dict[m].keys())
            ]
            lls_marg = [
                eval_dict[m][severity]["ll_marg"]
                for severity in sorted(eval_dict[m].keys())
            ]
            ents = [
                eval_dict[m][severity]["entropy"]
                for severity in sorted(eval_dict[m].keys())
            ]
            highest_class_probs = [
                eval_dict[m][severity]["highest_class_prob"]
                for severity in sorted(eval_dict[m].keys())
            ]
            # correct_class_probs = [
            #     eval_dict[m][severity]["correct_class_prob"]
            #     for severity in sorted(eval_dict[m].keys())
            # ]
            # dempster_shafers = [
            #     eval_dict[m][severity]["dempster_shafer"]
            #     for severity in sorted(eval_dict[m].keys())
            # ]

            fig, ax = plt.subplots()

            ax.set_xlabel("severity")
            ax.set_xticks(np.array(list(range(5))) + 1)

            ax.plot(backbone_acc, label="backbone acc", color="red")
            ax.plot(einet_acc, label="einet acc", color="orange")
            ax.set_ylabel("accuracy", color="red")
            ax.tick_params(axis="y", labelcolor="red")
            ax.set_ylim([0, 1])
            ax.grid(False)

            ax2 = ax.twinx()
            ax2.plot(lls_marg, label="ll", color="blue")
            ax2.tick_params(axis="y", labelcolor="blue")
            # ax2.set_ylim([0, 1])

            ax3 = ax.twinx()
            ax3.plot(ents, label="entropy", color="green")
            ax3.tick_params(axis="y", labelcolor="green")

            ax4 = ax.twinx()
            ax4.plot(highest_class_probs, label="highest class prob", color="purple")
            ax4.tick_params(axis="y", labelcolor="purple")

            # ax5 = ax.twinx()
            # ax5.plot(correct_class_probs, label="correct class prob", color="pink")
            # ax5.tick_params(axis="y", labelcolor="pink")

            # ax6 = ax.twinx()
            # ax6.plot(dempster_shafers, label="dempster shafer", color="black")
            # ax6.tick_params(axis="y", labelcolor="black")

            fig.legend(loc="upper left")
            fig.tight_layout()
            mlflow.log_figure(fig, f"{m}.png")
            plt.close()

        return lowest_val_loss
