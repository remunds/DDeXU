import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import mlflow


def load_svhn_test():
    from torchvision.datasets import SVHN
    from torchvision import transforms

    test_transformer = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            transforms.Lambda(lambda x: x.reshape(-1, 32 * 32 * 3).squeeze()),
        ]
    )

    test_ds = SVHN(
        root=dataset_dir + "svhn",
        split="test",
        download=True,
        transform=test_transformer,
    )

    return test_ds


dataset_dir = "/data_docker/datasets/"
cifar10_c_url = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"
cifar10_c_path = "CIFAR-10-C"
cifar10_c_path_complete = dataset_dir + cifar10_c_path


def load_datasets():
    # download cifar10-c
    if not os.path.exists(cifar10_c_path_complete + ".tar"):
        print("Downloading CIFAR-10-C...")
        os.system(f"wget {cifar10_c_url} -O {cifar10_c_path_complete}")

        print("Extracting CIFAR-10-C...")
        os.system(f"tar -xvf {cifar10_c_path_complete}.tar")

        print("Done!")

    # get normal cifar-10
    from torchvision.datasets import CIFAR10
    from torchvision import transforms

    train_transformer = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            transforms.Lambda(lambda x: x.reshape(-1, 32 * 32 * 3).squeeze()),
        ]
    )
    test_transformer = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            transforms.Lambda(lambda x: x.reshape(-1, 32 * 32 * 3).squeeze()),
        ]
    )

    train_ds = CIFAR10(
        root=dataset_dir + "cifar10",
        download=True,
        train=True,
        transform=train_transformer,
    )
    train_ds, valid_ds = torch.utils.data.random_split(
        train_ds, [45000, 5000], generator=torch.Generator().manual_seed(0)
    )
    test_ds = CIFAR10(
        root=dataset_dir + "cifar10",
        download=True,
        train=False,
        transform=test_transformer,
    )
    return train_ds, valid_ds, test_ds, test_transformer


def load_cifar10_c_test(test_ds):
    corruptions = [
        "brightness",
        "contrast",
        "defocus_blur",
        "elastic_transform",
        "fog",
        "frost",
        "gaussian_blur",
        "gaussian_noise",
        "glass_blur",
        "impulse_noise",
        "jpeg_compression",
        "motion_blur",
        "pixelate",
        "saturate",
        "shot_noise",
        "snow",
        "spatter",
        "speckle_noise",
        "zoom_blur",
    ]
    # available corrupted dataset: 50k*len(corruptions), 32, 32, 3
    levels = [1, 2, 3, 4, 5]
    datasets_length = 10000 * len(levels) * len(corruptions)
    all_data = np.zeros((datasets_length, 32, 32, 3), dtype=np.uint8)
    labels = np.zeros((datasets_length), dtype=np.uint8)
    for corr_idx, c in enumerate(corruptions):
        # each corrupted dataset has shape of test: 50k, 32, 32, 3
        data = np.load(f"{cifar10_c_path_complete}/{c}.npy")
        all_data[corr_idx * 10000 * 5 : (corr_idx + 1) * 10000 * 5] = data
        labels[corr_idx * 10000 * 5 : (corr_idx + 1) * 10000 * 5] = test_ds.labels
    print("done loading corruptions")
    return all_data, labels


def start_cifar10_calib_run(run_name, batch_sizes, model_params, train_params, trial):
    with mlflow.start_run(run_name=run_name):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlflow.log_param("device", device)

        ckpt_dir = f"/data_docker/ckpts/cifar10-c_calib/{run_name}/"
        mlflow.log_param("ckpt_dir", ckpt_dir)

        # log all params
        mlflow.log_params(batch_sizes)
        mlflow.log_params(model_params)
        mlflow.log_params(train_params)

        # load datasets
        train_ds, valid_ds, test_ds, test_transformer = load_datasets()
        batch_size = batch_sizes["resnet"]

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        valid_dl = DataLoader(
            valid_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        )

        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # Create model
        model_name = model_params["model"]
        del model_params["model"]
        if model_name == "ConvResNetDet":
            from Models import ConvResNetDet
            from torchvision.models.resnet import BasicBlock, Bottleneck

            if model_params["block"] == "basic":
                block = BasicBlock
            elif model_params["block"] == "bottleneck":
                block = Bottleneck
            else:
                raise NotImplementedError

            del model_params["block"]
            layers = model_params["layers"]
            del model_params["layers"]
            del model_params["spectral_normalization"]
            # del model_params["mod"]
            del model_params["num_hidden"]

            del model_params["train_batch_size"]
            del model_params["model_size"]

            model = ConvResNetDet(
                block,
                layers,
                explaining_vars=[],  # for calibration test, we don't need explaining vars
                **model_params,
            )
        elif model_name == "ConvResNetSPN":
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
            del model_params["num_hidden"]

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
        elif model_name == "EfficientNetDropout":
            from Models import EfficientNetDropout

            model = EfficientNetDropout(
                explaining_vars=[],  # for calibration test, we don't need explaining vars
                **model_params,
            )
        elif model_name == "EfficientNetDet":
            from Models import EfficientNetDet

            model = EfficientNetDet(
                explaining_vars=[],  # for calibration test, we don't need explaining vars
                **model_params,
            )
        elif model_name == "EfficientNetEnsemble":
            from Models import EfficientNetEnsemble

            model = EfficientNetEnsemble(
                explaining_vars=[],  # for calibration test, we don't need explaining vars
                **model_params,
            )
            for m in model.members:
                m.to(device)
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
        model.eval()
        # before costly evaluation, make sure that the model is not completely off
        model.deactivate_uncert_head()
        backbone_valid_acc = model.eval_acc(valid_dl, device)
        mlflow.log_metric("backbone_valid_acc", backbone_valid_acc)
        model.activate_uncert_head()

        # if valid_acc < 0.5:
        #     # let optuna know that this is a bad trial
        #     return lowest_val_loss
        if "GMM" in model_name:
            model.fit_gmm(train_dl, device)
        elif train_params["num_epochs"] > 0 or train_params["warmup_epochs"] > 0:
            mlflow.pytorch.log_state_dict(model.state_dict(), "model")

        # if train_params["num_epochs"] == 0:
        #     return lowest_val_loss

        # Evaluate
        eval_dict = {}

        # eval resnet
        model.deactivate_uncert_head()
        train_acc = model.eval_acc(train_dl, device)
        mlflow.log_metric("train accuracy backbone", train_acc)
        test_acc = model.eval_acc(test_dl, device)
        mlflow.log_metric("test accuracy backbone", test_acc)

        # eval einet
        model.activate_uncert_head()
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
        orig_test_ll_marg = model.eval_ll_marg(orig_test_ll, device, return_all=True)
        mlflow.log_metric("test_ll_marg", orig_test_ll_marg.mean().item())
        orig_test_pred_entropy = model.eval_entropy(
            orig_test_ll, device, return_all=True
        )
        mlflow.log_metric("test_entropy", torch.mean(orig_test_pred_entropy).item())

        # Calibration of test
        print("evaluating calibration")
        model.eval_calibration(
            orig_test_ll, device, "test", test_dl, method="posterior"
        )
        model.eval_calibration(orig_test_ll, device, "test", test_dl, method="entropy")
        model.eval_calibration(orig_test_ll, device, "test", test_dl, method="nll")

        # AUROC and AUPR for OOD detection vs SVHN
        svhn_test_ds = load_svhn_test()
        svhn_test_dl = DataLoader(
            svhn_test_ds,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=2,
        )
        print("Eval OOD SVHN")
        svhn_ll = model.eval_ll(svhn_test_dl, device, return_all=True)
        svhn_ll_marg = model.eval_ll_marg(svhn_ll, device, return_all=True)
        svhn_entropy = model.eval_entropy(svhn_ll, device, return_all=True)
        mlflow.log_metric("svhn_ll_marg", svhn_ll_marg.mean().item())
        mlflow.log_metric("svhn_entropy", torch.mean(svhn_entropy).item())

        (_, _, _), (_, _, _), auroc, auprc = model.eval_ood(
            orig_test_pred_entropy, svhn_entropy, device
        )
        mlflow.log_metric("auroc_svhn_entropy", auroc)
        mlflow.log_metric("auprc_svhn_entropy", auprc)

        (_, _, _), (_, _, _), auroc, auprc = model.eval_ood(
            orig_test_ll_marg, svhn_ll_marg, device, confidence=True
        )
        mlflow.log_metric("auroc_svhn_ll_marg", auroc)
        mlflow.log_metric("auprc_svhn_ll_marg", auprc)

        print("Done OOD SVHN")

        # train: 50k, 32, 32, 3
        # test: 10k, 32, 32, 3
        # test-corrupted: 10k, 32, 32, 3 per corruption level (5)

        corruptions = [
            "brightness",
            "contrast",
            "defocus_blur",
            "elastic_transform",
            "fog",  # was broken -> reload?
            "frost",
            "gaussian_blur",
            "gaussian_noise",
            "glass_blur",
            "impulse_noise",
            "jpeg_compression",
            "motion_blur",
            "pixelate",
            "saturate",
            "shot_noise",
            "snow",
            "spatter",
            "speckle_noise",
            "zoom_blur",
        ]
        from tqdm import tqdm

        print("loading all corrupted data")
        cifar10_c_ds = torch.zeros((10000 * len(corruptions) * 5, 32 * 32 * 3))
        index = 0
        for corruption in tqdm(corruptions):
            curr_cifar10 = np.load(f"{cifar10_c_path_complete}/{corruption}.npy")
            curr_cifar10 = torch.stack(
                [test_transformer(img) for img in curr_cifar10], dim=0
            )
            cifar10_c_ds[index : index + 10000 * 5] = curr_cifar10
            index += 10000 * 5
        targets = torch.cat(
            [torch.tensor(test_ds.targets) for _ in range(len(corruptions) * 5)], dim=0
        )

        print("shapes of corrupted stuff: ", cifar10_c_ds.shape, targets.shape)
        cifar10_c_ds = list(
            zip(
                cifar10_c_ds.to(dtype=torch.float32),
                targets.reshape(-1),
            )
        )

        cifar10_c_dl = DataLoader(
            cifar10_c_ds,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
        )

        # OOD vs CIFAR-10-C

        print("Eval OOD cifar10-c")
        cifar10_c_ll = model.eval_ll(cifar10_c_dl, device, return_all=True)
        cifar10_c_ll_marg = model.eval_ll_marg(cifar10_c_ll, device, return_all=True)
        mlflow.log_metric("cifar10_c_ll_marg", cifar10_c_ll_marg.mean().item())
        cifar10_c_entropy = model.eval_entropy(cifar10_c_ll, device, return_all=True)
        mlflow.log_metric("cifar10_c_entropy", torch.mean(cifar10_c_entropy).item())

        (_, _, _), (_, _, _), auroc, auprc = model.eval_ood(
            orig_test_pred_entropy, cifar10_c_entropy, device
        )
        mlflow.log_metric("auroc_cifar10_c_entropy", auroc)
        mlflow.log_metric("auprc_cifar10_c_entropy", auprc)

        (_, _, _), (_, _, _), auroc, auprc = model.eval_ood(
            orig_test_ll_marg, cifar10_c_ll_marg, device, confidence=True
        )
        mlflow.log_metric("auroc_cifar10_c_ll_marg", auroc)
        mlflow.log_metric("auprc_cifar10_c_ll_marg", auprc)

        print("Done OOD cifar10-c")

        # evaluate calibration
        print("evaluating calibration")
        model.eval_calibration(None, device, "test-c", cifar10_c_dl, method="posterior")
        model.eval_calibration(None, device, "test-c", cifar10_c_dl, method="entropy")
        model.eval_calibration(None, device, "test-c", cifar10_c_dl, method="nll")
        print("done evaluating calibration")

        del cifar10_c_ds, cifar10_c_dl

        # iterate over all corruptions, load dataset, evaluate
        for corruption in tqdm(corruptions):
            # load dataset
            data = np.load(f"{cifar10_c_path_complete}/{corruption}.npy")
            eval_dict[corruption] = {}
            # iterate over severity levels
            for severity in range(5):
                current_data = data[severity * 10000 : (severity + 1) * 10000]
                # transform with cifar10_transformer
                current_data = torch.stack(
                    [test_transformer(img) for img in current_data], dim=0
                )
                corrupt_test_ds = list(
                    zip(
                        current_data,
                        test_ds.targets,
                    )
                )
                test_dl = DataLoader(
                    corrupt_test_ds,
                    batch_size=batch_size,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=1,
                )

                # evaluate
                model.deactivate_uncert_head()
                backbone_acc = model.eval_acc(test_dl, device)
                model.activate_uncert_head()
                acc = model.eval_acc(test_dl, device)
                test_ll = model.eval_ll(test_dl, device, return_all=True)
                test_ll_marg = model.eval_ll_marg(test_ll, device)
                test_entropy = model.eval_entropy(test_ll, device)
                highest_class_prob = model.eval_highest_class_prob(test_ll, device)
                eval_dict[corruption][severity] = {
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

        from plotting_utils import uncert_corrupt_plot

        all_corruption_accs = []
        all_corruption_lls = []
        all_corruption_entropies = []
        for corruption in eval_dict:
            backbone_accs = [
                eval_dict[corruption][severity]["backbone_acc"]
                for severity in eval_dict[corruption]
            ]
            # shape: (severity)
            all_corruption_accs.append(backbone_accs)
            # einet_accs = [
            #     eval_dict[corruption][severity]["einet_acc"]
            #     for severity in eval_dict[corruption]
            # ]
            lls_marg = [
                eval_dict[corruption][severity]["ll_marg"]
                for severity in eval_dict[corruption]
            ]
            all_corruption_lls.append(lls_marg)
            entropy = [
                eval_dict[corruption][severity]["entropy"]
                for severity in eval_dict[corruption]
            ]
            all_corruption_entropies.append(entropy)

            uncert_corrupt_plot(
                backbone_accs,
                lls_marg,
                f"{corruption}",
                mode="ll",
            )
            uncert_corrupt_plot(
                backbone_accs,
                entropy,
                f"{corruption}",
                mode="entropy",
            )
        # shapes: (corruptions, severity)
        all_accs = np.array(all_corruption_accs)
        all_lls = np.array(all_corruption_lls)
        all_entropies = np.array(all_corruption_entropies)

        fig = uncert_corrupt_plot(
            all_accs.mean(axis=0),
            all_lls.mean(axis=0),
            f"All corruptions",
            mode="ll",
        )
        mlflow.log_figure(fig, "all_corruptions_ll.pdf")
        fig = uncert_corrupt_plot(
            all_accs.mean(axis=0),
            all_entropies.mean(axis=0),
            f"All corruptions",
            mode="entropy",
        )
        mlflow.log_figure(fig, "all_corruptions_entropy.pdf")

        return lowest_val_loss
