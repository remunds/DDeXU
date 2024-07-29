import os
import numpy as np
from torch.utils.data import DataLoader
import torch
import mlflow
from torchvision.datasets import SVHN


dataset_dir = "/data_docker/datasets/"
svhn_c_path = "svhn_c"
svhn_c_path_complete = dataset_dir + svhn_c_path
corruptions = [
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
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


def load_datasets():
    # get normal svhn
    from torchvision import transforms

    train_transformer = transforms.Compose(
        [
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.198, 0.201, 0.197)),
            # transforms.Lambda(lambda x: x.reshape(-1, 32 * 32 * 3).squeeze()),
        ]
    )
    test_transformer = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.198, 0.201, 0.197)),
            # transforms.Lambda(lambda x: x.reshape(-1, 32 * 32 * 3).squeeze()),
        ]
    )
    train_ds = SVHN(
        root=dataset_dir + "svhn",
        download=True,
        split="train",
        # transform=svhn_transformer,
    )
    train_data = [train_transformer(img).flatten() for img, _ in train_ds]
    train_data = torch.concat(
        [
            torch.zeros((train_ds.data.shape[0], len(corruptions))),
            torch.stack(train_data, dim=0),
        ],
        dim=1,
    )
    train_data = list(zip(train_data, train_ds.labels))

    test_ds = SVHN(
        root=dataset_dir + "svhn",
        download=True,
        split="test",
        # transform=svhn_transformer,
    )

    return train_data, test_ds, test_transformer


def get_corrupted_svhn(
    all_corruption_len: int, corruptions: list, test_labels: np.ndarray, levels: list
):
    num_samples = 26032
    # available corrupted dataset: 26032*len(corruptions), 32, 32, 3
    datasets_length = num_samples * len(levels) * len(corruptions) // 2
    train_corrupt_data = np.zeros((datasets_length, 32, 32, 3), dtype=np.uint8)
    test_corrupt_data = np.zeros((datasets_length, 32, 32, 3), dtype=np.uint8)
    train_corrupt_levels = np.zeros(
        (datasets_length, all_corruption_len), dtype=np.uint8
    )
    test_corrupt_levels = np.zeros(
        (datasets_length, all_corruption_len), dtype=np.uint8
    )
    train_corrupt_labels = np.zeros((datasets_length), dtype=np.uint8)
    # test_corrupt_labels = np.zeros((datasets_length) + 5000, dtype=np.uint8)
    test_corrupt_labels = np.zeros((datasets_length), dtype=np.uint8)
    train_idx = 0
    test_idx = 0
    for corr_idx, c in enumerate(corruptions):
        # each corrupted dataset has shape of test: 26032, 32, 32, 3
        for i in range(5):  # iterate over corruption levels
            if not i in levels:
                continue
            data = np.load(f"{svhn_c_path_complete}/svhn_test_{c}_l{i+1}.npy")
            data_idx = i * num_samples  # not sure

            new_train_idx = train_idx + num_samples // 2
            new_test_idx = test_idx + num_samples // 2
            train_corrupt_data[train_idx:new_train_idx] = data[: num_samples // 2, ...]
            train_corrupt_levels[train_idx:new_train_idx, corr_idx] = i + 1
            train_corrupt_labels[train_idx:new_train_idx] = test_labels[
                : num_samples // 2
            ]

            test_corrupt_data[test_idx:new_test_idx] = data[num_samples // 2 :, ...]
            test_corrupt_levels[test_idx:new_test_idx, corr_idx] = i + 1
            test_corrupt_labels[test_idx:new_test_idx] = test_labels[num_samples // 2 :]
            train_idx = new_train_idx
            test_idx = new_test_idx

    print("done loading corruptions")
    return (
        train_corrupt_data,
        train_corrupt_levels,
        train_corrupt_labels,
        test_corrupt_data,
        test_corrupt_levels,
        test_corrupt_labels,
    )


def start_svhn_expl_run(run_name, batch_sizes, model_params, train_params, trial):
    with mlflow.start_run(run_name=run_name) as run:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlflow.log_param("device", device)

        ckpt_dir = f"/data_docker/ckpts/svhn-c_expl/{run_name}/"
        mlflow.log_param("ckpt_dir", ckpt_dir)

        # log all params
        mlflow.log_params(batch_sizes)
        mlflow.log_params(model_params)
        mlflow.log_params(train_params)

        # load data
        train_data, test_ds, test_transformer = load_datasets()

        levels = train_params["corruption_levels_train"]
        del train_params["corruption_levels_train"]
        if type(levels) != list:
            raise ValueError("corruption_levels must be a list")

        print("loading train corruption data")
        (
            train_corrupt_data,
            train_corrupt_levels,
            train_corrupt_labels,
            _,
            _,
            _,
        ) = get_corrupted_svhn(
            # yes, test_ds is correct here
            len(corruptions),
            corruptions,
            np.array(test_ds.labels),
            levels,
        )
        print("done loading corrupted data")
        # use the test_transformer -> no augmentation
        train_corrupt_data = [
            test_transformer(img).flatten() for img in train_corrupt_data
        ]
        train_corrupt_data = torch.concat(
            [
                torch.from_numpy(train_corrupt_levels).to(dtype=torch.int32),
                torch.stack(train_corrupt_data, dim=0).to(dtype=torch.float32),
            ],
            dim=1,
        )

        train_corrupt_data = list(zip(train_corrupt_data, train_corrupt_labels))
        # We want to train on the corrupted data, s.t. explanations are possible

        train_ds, valid_ds = torch.utils.data.random_split(
            train_corrupt_data, [0.9, 0.1], generator=torch.Generator().manual_seed(0)
        )
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_sizes["resnet"],
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )

        valid_dl = DataLoader(
            valid_ds,
            batch_size=batch_sizes["resnet"],
            shuffle=True,
            pin_memory=True,
            num_workers=1,
        )

        print("done loading data")

        # Create model
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
                **model_params,
            )
        elif model_name == "AutoEncoderSPN":
            from Models import AutoEncoderSPN

            model = AutoEncoderSPN(
                **model_params,
            )
        elif model_name == "EfficientNetSPN":
            from Models import EfficientNetSPN

            model = EfficientNetSPN(
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
        model.compute_normalization_values(train_dl, device)
        model.deactivate_uncert_head()
        valid_acc = model.eval_acc(valid_dl, device)
        mlflow.log_metric("backbone_valid_acc", valid_acc)
        model.activate_uncert_head()

        if "GMM" in model_name:
            model.fit_gmm(train_dl, device)
        elif train_params["num_epochs"] > 0 or train_params["warmup_epochs"] > 0:
            mlflow.pytorch.log_state_dict(model.state_dict(), "model")

        # if train_params["num_epochs"] == 0:
        #     return lowest_val_loss

        # Evaluate
        model.eval()

        valid_ll_marg = model.eval_ll_marg(
            None,
            device,
            valid_dl,
        )
        mlflow.log_metric("valid_ll_marg", valid_ll_marg)

        # test with all corruption-levels
        test_levels = [0, 1, 2, 3, 4]
        print("loading test corrupted data")
        (
            _,
            _,
            _,
            test_corrupt_data,
            test_corrupt_levels,
            test_corrupt_labels,
        ) = get_corrupted_svhn(
            len(corruptions), corruptions, np.array(test_ds.labels), test_levels
        )
        print("done loading test corrupted data")

        test_corrupt_data = [
            test_transformer(img).flatten() for img in test_corrupt_data
        ]
        test_corrupt_data = torch.concat(
            [
                torch.from_numpy(test_corrupt_levels).to(dtype=torch.int32),
                torch.stack(test_corrupt_data, dim=0).to(dtype=torch.float32),
            ],
            dim=1,
        )
        print("test_corrupt_data.shape: ", test_corrupt_data.shape)
        print("test_corrupt_labels.shape: ", test_corrupt_labels.shape)

        # test_corrupt_levels.shape = [num_data_points, num_corruptions]
        from tqdm import tqdm

        eval_dict = {}

        for corr_idx, corruption in tqdm(enumerate(corruptions)):
            eval_dict[corruption] = {}
            for corr_level in test_levels:
                eval_dict[corruption][corr_level] = {}
                data_idxs = test_corrupt_levels[:, corr_idx] == corr_level + 1
                data = test_corrupt_data[data_idxs]
                labels = test_corrupt_labels[data_idxs]

                ds = list(zip(data, labels))
                dl = DataLoader(
                    ds,
                    batch_size=batch_sizes["resnet"],
                    shuffle=False,
                    pin_memory=False,
                    # pin_memory=True,
                    # num_workers=1,
                )
                acc = model.eval_acc(dl, device)
                eval_dict[corruption][corr_level]["acc"] = acc
                log_p_x_g_y = model.eval_ll(dl, device, return_all=True)
                entropy = model.eval_entropy(log_p_x_g_y, device)
                eval_dict[corruption][corr_level]["entropy"] = entropy
                ll_marg = model.eval_ll_marg(log_p_x_g_y, device)
                eval_dict[corruption][corr_level]["ll_marg"] = ll_marg
                expl_ll = model.explain_ll(dl, device)
                # eval_dict[corruption][corr_level]["expl_ll"] = expl_ll[corr_idx]
                eval_dict[corruption][corr_level][
                    "expl_ll"
                ] = expl_ll  # len: [1, num_expl_vars]
                expl_mpe = model.explain_mpe(dl, device)
                eval_dict[corruption][corr_level]["expl_mpe"] = expl_mpe.tolist()

                expl_post = model.explain_posterior(dl, device)
                eval_dict[corruption][corr_level]["expl_post"] = expl_post

        mlflow.log_dict(eval_dict, "eval_dict")

        overall_acc = np.mean(
            [eval_dict[c][l]["acc"] for c in eval_dict for l in eval_dict[c]]
        )
        mlflow.log_metric("manip_acc", overall_acc)
        overall_ll_marg = np.mean(
            [eval_dict[c][l]["ll_marg"] for c in eval_dict for l in eval_dict[c]]
        )
        mlflow.log_metric("manip_ll_marg", overall_ll_marg)
        overall_entropy = np.mean(
            [eval_dict[c][l]["entropy"] for c in eval_dict for l in eval_dict[c]]
        )
        mlflow.log_metric("manip_entropy", overall_entropy)

        from plotting_utils import explain_plot

        # as in calib_experiment
        for corruption in eval_dict:
            # accs = [eval_dict[corruption][l]["acc"] for l in eval_dict[corruption]]
            lls_marg = [
                eval_dict[corruption][l]["ll_marg"] for l in eval_dict[corruption]
            ]
            entropy = [
                eval_dict[corruption][l]["entropy"] for l in eval_dict[corruption]
            ]
            # get corruption index
            corr_idx = corruptions.index(corruption)
            expl_ll = [
                eval_dict[corruption][l]["expl_ll"] for l in eval_dict[corruption]
            ]  # list of list, shape: [num_levels, num_expl_vars]
            expl_ll = torch.tensor(expl_ll)
            expl_mpe = [
                eval_dict[corruption][l]["expl_mpe"] for l in eval_dict[corruption]
            ]
            expl_mpe = torch.tensor(expl_mpe)

            expl_post = [
                eval_dict[corruption][l]["expl_post"] for l in eval_dict[corruption]
            ]
            expl_post = torch.tensor(expl_post)

            fig = explain_plot(corruptions, lls_marg, expl_ll, corruption, "ll")
            mlflow.log_figure(fig, f"ll_{corruption}.pdf")
            fig = explain_plot(corruptions, entropy, expl_mpe, corruption, "mpe")
            mlflow.log_figure(fig, f"mpe_{corruption}.pdf")
            fig = explain_plot(corruptions, entropy, expl_post, corruption, "post")
            mlflow.log_figure(fig, f"post_{corruption}.pdf")

            # expl_ll.shape = [num_levels, num_expl_vars]

        return lowest_val_loss
