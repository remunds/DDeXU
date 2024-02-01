import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import mlflow


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


def start_cifar10_calib_run_gmm(
    run_name, batch_sizes, model_params, train_params, trial
):
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
        if model_name == "EfficientNetGMM":
            from ResNetSPN import EfficientNetGMM

            model = EfficientNetGMM(
                explaining_vars=[],  # for calibration test, we don't need explaining vars
                **model_params,
            )
        elif model_name == "ConvResnetDDUGMM":
            from ResNetSPN import ConvResnetDDUGMM

            model = ConvResnetDDUGMM(
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
        print("done backbone training, fit gmm")
        model.fit_gmm(train_dl, device)
        print("done fitting gmm")

        # Evaluate
        model.eval()
        eval_dict = {}

        # eval resnet
        model.gmm_active = False
        train_acc = model.eval_acc(train_dl, device)
        mlflow.log_metric("train accuracy backbone", train_acc)
        test_acc = model.eval_acc(test_dl, device)
        mlflow.log_metric("test accuracy backbone", test_acc)

        # eval gmm
        model.gmm_active = True
        train_acc = model.eval_acc(train_dl, device)
        mlflow.log_metric("train_acc", train_acc)
        train_ll = model.eval_ll(train_dl, device)
        mlflow.log_metric("train_ll", train_ll)
        train_pred_entropy = model.eval_entropy(train_dl, device)
        mlflow.log_metric("train_entropy", train_pred_entropy)

        test_acc = model.eval_acc(test_dl, device)
        mlflow.log_metric("test_acc", test_acc)
        orig_test_ll = model.eval_ll(test_dl, device)
        mlflow.log_metric("test_ll", orig_test_ll)
        orig_test_pred_entropy = model.eval_entropy(test_dl, device)
        mlflow.log_metric("test_entropy", orig_test_pred_entropy)

        # random noise baseline
        random_data = np.random.rand(10000, 32, 32, 3)
        random_data = torch.stack([test_transformer(img) for img in random_data], dim=0)
        random_ds = list(zip(random_data.to(dtype=torch.float32), test_ds.targets))
        random_dl = DataLoader(
            random_ds,
            batch_size=batch_sizes["resnet"],
            shuffle=False,
            pin_memory=True,
            num_workers=1,
        )
        random_acc = model.eval_acc(random_dl, device)
        mlflow.log_metric("random_acc", random_acc)
        random_ll = model.eval_ll(random_dl, device)
        mlflow.log_metric("random_ll", random_ll)
        random_pred_entropy = model.eval_entropy(random_dl, device)
        mlflow.log_metric("random_entropy", random_pred_entropy)

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
            batch_size=batch_sizes["resnet"],
            shuffle=False,
            pin_memory=True,
            num_workers=2,
        )

        # evaluate calibration
        print("evaluating calibration")
        model.eval_calibration(cifar10_c_dl, device, name="cifar10-c", n_bins=20)
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
                    batch_size=batch_sizes["resnet"],
                    shuffle=False,
                    pin_memory=True,
                    num_workers=1,
                )

                # evaluate
                model.gmm_active = False
                backbone_acc = model.eval_acc(test_dl, device)
                model.gmm_active = True
                acc = model.eval_acc(test_dl, device)
                # test_ll = model.eval_ll(test_dl, device)
                # rewrote to use logsumexp like DDU
                test_ll = model.eval_ll_unmarginalized(test_dl, device, return_all=True)
                test_ll = torch.logsumexp(test_ll, dim=1).mean().item()
                test_entropy = model.eval_entropy(test_dl, device)
                highest_class_prob = model.eval_highest_class_prob(test_dl, device)

                eval_dict[corruption][severity] = {
                    "backbone_acc": backbone_acc,
                    "gmm_acc": acc,
                    "ll": test_ll,
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

        gmm_acc = np.mean(
            [
                eval_dict[corruption][severity]["gmm_acc"]
                for corruption in eval_dict
                for severity in eval_dict[corruption]
            ]
        )

        ll = np.mean(
            [
                eval_dict[corruption][severity]["ll"]
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

        mlflow.log_metric("manip_gmm_acc", gmm_acc)
        mlflow.log_metric("manip_backbone_acc", backbone_acc)
        mlflow.log_metric("manip_ll", ll)
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
            gmm_accs = [
                eval_dict[corruption][severity]["gmm_acc"]
                for severity in eval_dict[corruption]
            ]
            lls = [
                eval_dict[corruption][severity]["ll"]
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
            ax.plot(gmm_accs, label="gmm acc", color="orange")
            ax.set_ylabel("accuracy", color="red")
            ax.tick_params(axis="y", labelcolor="red")
            ax.set_ylim([0, 1])

            ax2 = ax.twinx()
            ax2.plot(lls, label="ll", color="blue")
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
