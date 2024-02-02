import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import mlflow


dataset_dir = "/data_docker/datasets/"
svhn_c_path = "svhn_c"
svhn_c_path_complete = dataset_dir + svhn_c_path


def load_datasets():
    # get normal svhn
    from torchvision.datasets import SVHN
    from torchvision import transforms

    svhn_transformer = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.198, 0.201, 0.197)),
            transforms.Lambda(lambda x: x.reshape(-1, 32 * 32 * 3).squeeze()),
        ]
    )

    train_ds = SVHN(
        root=dataset_dir + "svhn",
        download=True,
        split="train",
        transform=svhn_transformer,
    )
    train_ds, valid_ds = torch.utils.data.random_split(
        train_ds, [0.8, 0.2], generator=torch.Generator().manual_seed(0)
    )
    test_ds = SVHN(
        root=dataset_dir + "svhn",
        download=True,
        split="test",
        transform=svhn_transformer,
    )

    return train_ds, valid_ds, test_ds, svhn_transformer


def start_svhn_calib_run(run_name, batch_sizes, model_params, train_params, trial):
    with mlflow.start_run(run_name=run_name):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlflow.log_param("device", device)

        ckpt_dir = f"/data_docker/ckpts/svhn-c_calib/{run_name}/"
        mlflow.log_param("ckpt_dir", ckpt_dir)

        # log all params
        mlflow.log_params(batch_sizes)
        mlflow.log_params(model_params)
        mlflow.log_params(train_params)

        # load datasets
        train_ds, valid_ds, test_ds, svhn_transformer = load_datasets()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_sizes["resnet"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        valid_dl = DataLoader(
            valid_ds,
            batch_size=batch_sizes["resnet"],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        )

        test_dl = DataLoader(test_ds, batch_size=batch_sizes["resnet"], shuffle=False)

        # Create model
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
        elif model_name == "EfficientNetGMM":
            from ResNetSPN import EfficientNetGMM

            model = EfficientNetGMM(
                explaining_vars=[],  # for calibration test, we don't need explaining vars
                **model_params,
            )
        elif model_name == "ConvResnetDDUGMM":
            from ResNetSPN import ConvResnetDDUGMM
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
        if "GMM" in model_name:
            model.fit_gmm(train_dl, device)
        else:
            mlflow.pytorch.log_state_dict(model.state_dict(), "model")

        # Evaluate
        model.eval()
        eval_dict = {}

        # eval einet
        model.einet_active = True
        train_acc = model.eval_acc(train_dl, device)
        mlflow.log_metric("train_acc", train_acc)
        train_ll = model.eval_ll(train_dl, device, return_all=True)
        train_ll_marg = model.eval_ll_marg(train_ll, device)
        mlflow.log_metric("train_ll_marg", train_ll_marg)
        train_pred_entropy = model.eval_entropy(train_ll, device)
        mlflow.log_metric("train_entropy", train_pred_entropy)

        test_acc = model.eval_acc(test_dl, device)
        mlflow.log_metric("test_acc", test_acc)
        orig_test_ll = model.eval_ll(test_dl, device, return_all=True)
        orig_test_ll_marg = model.eval_ll_marg(orig_test_ll, device)
        mlflow.log_metric("test_ll_marg", orig_test_ll_marg)
        orig_test_pred_entropy = model.eval_entropy(orig_test_ll, device)
        mlflow.log_metric("test_entropy", orig_test_pred_entropy)

        # random noise baseline
        random_data = np.random.rand(10000, 32, 32, 3)
        random_data = torch.stack([svhn_transformer(img) for img in random_data], dim=0)
        random_ds = list(zip(random_data.to(dtype=torch.float32), test_ds.labels))
        random_dl = DataLoader(
            random_ds,
            batch_size=batch_sizes["resnet"],
            shuffle=False,
            pin_memory=True,
            num_workers=1,
        )
        random_acc = model.eval_acc(random_dl, device)
        mlflow.log_metric("random_acc", random_acc)
        random_ll = model.eval_ll(random_dl, device, return_all=True)
        random_ll_marg = model.eval_ll_marg(random_ll, device)
        mlflow.log_metric("random_ll_marg", random_ll_marg)
        random_pred_entropy = model.eval_entropy(random_ll, device)
        mlflow.log_metric("random_entropy", random_pred_entropy)

        # evaluate calibration
        print("evaluating calibration")
        model.eval_calibration(orig_test_ll, device, name="svhn", dl=test_dl)
        print("done evaluating calibration")

        # test: 26032, 32, 32, 3
        # test-corrupted: 26032, 32, 32, 3 per corruption level (5)

        corruptions = [
            "brightness",
            "contrast",
            "defocus_blur",
            "elastic_transform",
            "fog",  # was broken -> reload?
            "frost",
            # "gaussian_blur",
            "gaussian_noise",
            "glass_blur",
            "impulse_noise",
            "jpeg_compression",
            "motion_blur",
            "pixelate",
            # "saturate",
            "shot_noise",
            "snow",
            # "spatter",
            # "speckle_noise",
            "zoom_blur",
        ]
        from tqdm import tqdm

        levels = [1, 2, 3, 4, 5]
        num_samples = 26032

        print("loading all corrupted data")
        svhn_c_ds = torch.zeros((num_samples * len(corruptions) * 5, 32 * 32 * 3))
        index = 0
        for corruption in tqdm(corruptions):
            for l in levels:
                curr_svhn = np.load(
                    f"{svhn_c_path_complete}/svhn_test_{corruption}_l{l}.npy"
                )
                curr_svhn = torch.stack(
                    [svhn_transformer(img) for img in curr_svhn], dim=0
                )
                svhn_c_ds[index : index + num_samples] = curr_svhn
                index += num_samples
        targets = torch.cat(
            [torch.tensor(test_ds.labels) for _ in range(len(corruptions) * 5)], dim=0
        )

        print("shapes of corrupted stuff: ", svhn_c_ds.shape, targets.shape)
        svhn_c_ds = list(
            zip(
                svhn_c_ds.to(dtype=torch.float32),
                targets.reshape(-1),
            )
        )

        svhn_c_dl = DataLoader(
            svhn_c_ds,
            batch_size=batch_sizes["resnet"],
            shuffle=False,
            pin_memory=True,
            num_workers=2,
        )

        # evaluate calibration
        print("evaluating calibration on corrupted data")
        model.eval_calibration(None, device, name="svhn-c", dl=svhn_c_dl)
        print("done evaluating calibration on corrupted data")

        del svhn_c_ds, svhn_c_dl

        # iterate over all corruptions, load dataset, evaluate
        for corruption in tqdm(corruptions):
            # load dataset
            eval_dict[corruption] = {}
            # iterate over severity levels
            for l in levels:
                current_data = np.load(
                    f"{svhn_c_path_complete}/svhn_test_{corruption}_l{l}.npy"
                )
                # transform with svhn_transformer
                current_data = torch.stack(
                    [svhn_transformer(img) for img in current_data], dim=0
                )
                corrupt_test_ds = list(
                    zip(
                        current_data,
                        test_ds.labels,
                    )
                )
                test_dl = DataLoader(
                    corrupt_test_ds,
                    batch_size=batch_sizes["resnet"],
                    shuffle=False,
                    pin_memory=True,
                    num_workers=1,
                )

                # evaluate
                model.einet_active = False
                backbone_acc = model.eval_acc(test_dl, device)
                model.einet_active = True
                acc = model.eval_acc(test_dl, device)
                test_ll = model.eval_ll(test_dl, device, return_all=True)
                test_ll_marg = model.eval_ll_marg(test_ll, device)
                test_entropy = model.eval_entropy(test_ll, device)
                highest_class_prob = model.eval_highest_class_prob(test_ll, device)

                eval_dict[corruption][l] = {
                    "backbone_acc": backbone_acc,
                    "einet_acc": acc,
                    "ll_marg": test_ll_marg,
                    "entropy": test_entropy,
                    "highest_class_prob": highest_class_prob,
                }
        mlflow.log_dict(eval_dict, "eval_dict")

        backbone_acc = np.mean(
            [
                eval_dict[corruption][severity]["backbone_acc"]
                for corruption in eval_dict
                for severity in eval_dict[corruption]
            ]
        )

        einet_acc = np.mean(
            [
                eval_dict[corruption][severity]["einet_acc"]
                for corruption in eval_dict
                for severity in eval_dict[corruption]
            ]
        )

        ll_marg = np.mean(
            [
                eval_dict[corruption][severity]["ll_marg"]
                for corruption in eval_dict
                for severity in eval_dict[corruption]
            ]
        )

        entropy = np.mean(
            [
                eval_dict[corruption][severity]["entropy"]
                for corruption in eval_dict
                for severity in eval_dict[corruption]
            ]
        )

        highest_class_prob = np.mean(
            [
                eval_dict[corruption][severity]["highest_class_prob"]
                for corruption in eval_dict
                for severity in eval_dict[corruption]
            ]
        )

        mlflow.log_metric("manip_einet_acc", einet_acc)
        mlflow.log_metric("manip_backbone_acc", backbone_acc)
        mlflow.log_metric("manip_ll_marg", ll_marg)
        mlflow.log_metric("manip_entropy", entropy)
        mlflow.log_metric("manip_highest_class_prob", highest_class_prob)

        # create plot for each corruption
        # x axis: severity
        # y axis: acc, ll

        import matplotlib.pyplot as plt

        for corruption in eval_dict:
            backbone_accs = [
                eval_dict[corruption][severity]["backbone_acc"]
                for severity in eval_dict[corruption]
            ]
            einet_accs = [
                eval_dict[corruption][severity]["einet_acc"]
                for severity in eval_dict[corruption]
            ]
            lls_marg = [
                eval_dict[corruption][severity]["ll_marg"]
                for severity in eval_dict[corruption]
            ]
            entropy = [
                eval_dict[corruption][severity]["entropy"]
                for severity in eval_dict[corruption]
            ]

            fig, ax = plt.subplots()
            ax.set_xlabel("severity")
            ax.set_xticks(np.array(list(range(5))) + 1)

            ax.plot(backbone_accs, label="backbone acc", color="red")
            ax.plot(einet_accs, label="einet acc", color="orange")
            ax.set_ylabel("accuracy", color="red")
            ax.tick_params(axis="y", labelcolor="red")
            ax.set_ylim([0, 1])

            ax2 = ax.twinx()
            ax2.plot(lls_marg, label="ll", color="blue")
            ax2.tick_params(axis="y", labelcolor="blue")

            ax3 = ax.twinx()
            ax3.plot(entropy, label="entropy", color="green")
            ax3.tick_params(axis="y", labelcolor="green")

            ax4 = ax.twinx()
            ax4.plot(highest_class_prob, label="highest class prob", color="purple")
            ax4.tick_params(axis="y", labelcolor="purple")

            fig.legend(loc="upper left")
            fig.tight_layout()
            mlflow.log_figure(fig, f"{corruption}.png")
            plt.close()

        return lowest_val_loss
