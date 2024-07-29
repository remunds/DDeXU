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
    "gaussian_noise",
    # "brightness",
    # "contrast",
    # "defocus_blur",
    # "elastic_transform",
    # "fog",
    "frost",
    # "gaussian_blur", # not available
    # "glass_blur",
    # "impulse_noise",
    # "jpeg_compression",
    "motion_blur",
    # "pixelate",
    # "saturate", # not available
    # "shot_noise",
    # "snow",
    # "spatter", # not available
    # "speckle_noise", # not available
    # "zoom_blur",
]

# Train on modified (strength 1-2) data (feed modification strength to explaining variables)

# Experiment 1: LL Explanations (Explaining variables available)
# Evaluate on strength=5 of known variables
# Expectation: low LL (high epistemic uncertainty), high marginal LLs for the known variables

# Experiment 2: MPE Explanations (Explaining variables unavailable)
# Evaluate on strength=[1, 2] of variables that we do not know at inference time
# Expectation: MPEs reflect the actual value of the explaining variables


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
            torch.zeros((train_ds.data.shape[0], len(corruptions)), dtype=torch.int32),
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

    print("train-samples: ", len(train_ds))

    return train_data, train_ds, test_ds, test_transformer


# def get_corrupted_svhn_train(all_corruptions: list, corruptions: list, levels: list):
#     num_samples = 73257
#     # available corrupted dataset: 73257*len(corruptions), 32, 32, 3
#     datasets_length = num_samples * len(levels) * len(corruptions)
#     train_corrupt_data = np.zeros((datasets_length, 32, 32, 3), dtype=np.uint8)
#     train_corrupt_levels = np.zeros(
#         (datasets_length, len(all_corruptions)), dtype=np.uint8
#     )
#     train_idx = 0
#     for corr_idx, c in enumerate(all_corruptions):
#         if c not in corruptions:
#             continue
#         # each corrupted dataset has shape of train: 73257, 32, 32, 3
#         for i in range(5):  # iterate over corruption levels
#             if not i in levels:
#                 continue
#             data = np.load(f"{svhn_c_path_complete}/svhn_train_{c}_l{i+1}.npy")

#             new_train_idx = train_idx + num_samples
#             train_corrupt_data[train_idx:new_train_idx] = data[:num_samples, ...]
#             train_corrupt_levels[train_idx:new_train_idx, corr_idx] = i + 1

#             train_idx = new_train_idx

#     print("done loading train corruptions")
#     return (
#         train_corrupt_data,
#         train_corrupt_levels,
#     )


def get_corrupted_svhn_train(all_corruptions: list, corruptions: list, levels: list):
    num_samples = int(26032 / 2)
    # available corrupted dataset: 26032*len(corruptions), 32, 32, 3
    datasets_length = num_samples * len(levels) * len(corruptions)
    test_corrupt_data = np.zeros((datasets_length, 32, 32, 3), dtype=np.uint8)
    test_corrupt_levels = np.zeros(
        (datasets_length, len(all_corruptions)), dtype=np.uint8
    )
    test_idx = 0
    for corr_idx, c in enumerate(all_corruptions):
        if c not in corruptions:
            continue
        # each corrupted dataset has shape of test: 26032, 32, 32, 3
        for i in range(5):  # iterate over corruption levels
            if not i in levels:
                continue
            data = np.load(f"{svhn_c_path_complete}/svhn_test_{c}_l{i+1}.npy")

            new_test_idx = test_idx + num_samples
            test_corrupt_data[test_idx:new_test_idx] = data[:num_samples, ...]
            test_corrupt_levels[test_idx:new_test_idx, corr_idx] = i + 1

            test_idx = new_test_idx

    print("done loading test corruptions")
    return (
        test_corrupt_data,
        test_corrupt_levels,
    )


def get_corrupted_svhn_test(all_corruptions: list, corruptions: list, levels: list):
    # NOTE: this is only half the test set as we use it for training, too
    num_samples = int(26032 / 2)
    # available corrupted dataset: 26032*len(corruptions), 32, 32, 3
    datasets_length = num_samples * len(levels) * len(corruptions)
    test_corrupt_data = np.zeros((datasets_length, 32, 32, 3), dtype=np.uint8)
    test_corrupt_levels = np.zeros(
        (datasets_length, len(all_corruptions)), dtype=np.uint8
    )
    test_idx = 0
    for corr_idx, c in enumerate(all_corruptions):
        if c not in corruptions:
            continue
        # each corrupted dataset has shape of test: 26032, 32, 32, 3
        for i in range(5):  # iterate over corruption levels
            if not i in levels:
                continue
            data = np.load(f"{svhn_c_path_complete}/svhn_test_{c}_l{i+1}.npy")

            new_test_idx = test_idx + num_samples
            test_corrupt_data[test_idx:new_test_idx] = data[
                num_samples : num_samples * 2, ...
            ]
            test_corrupt_levels[test_idx:new_test_idx, corr_idx] = i + 1

            test_idx = new_test_idx

    print("done loading test corruptions")
    return (
        test_corrupt_data,
        test_corrupt_levels,
    )


def start_svhn_expl_run(run_name, batch_sizes, model_params, train_params, trial):
    run_name = run_name + f"_pres"
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
        train_data, train_ds_orig, test_ds, test_transformer = load_datasets()

        # levels = train_params["corruption_levels_train"]
        del train_params["corruption_levels_train"]
        # if type(levels) != list:
        #     raise ValueError("corruption_levels must be a list")

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

        # Train on level 1-2 corrupted data
        levels = [0, 1]
        train_data_corrupt, train_levels = get_corrupted_svhn_train(
            corruptions, corruptions, levels
        )
        train_data_corrupt = [
            test_transformer(img).flatten() for img in train_data_corrupt
        ]
        train_data_corrupt = torch.concat(
            [
                torch.from_numpy(train_levels).to(dtype=torch.int32),
                torch.stack(train_data_corrupt, dim=0).to(dtype=torch.float32),
            ],
            dim=1,
        )
        print("train corrupt shape: ", train_data_corrupt.shape)
        labels = [
            test_ds.labels[: int(len(test_ds.labels) / 2)]
            for i in range(len(levels))
            for j in range(len(corruptions))
        ]
        labels = np.concatenate(labels)
        print("labels.shape: ", labels.shape)
        train_data_corrupt = list(zip(train_data_corrupt, labels))
        train_ds, valid_ds = torch.utils.data.random_split(
            train_data_corrupt, [0.8, 0.2], generator=torch.Generator().manual_seed(0)
        )
        print(train_ds[0][0])
        print(valid_ds[0][0])
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

        print("training backbone")

        backup_epochs = train_params["num_epochs"]
        train_params["num_epochs"] = 0

        # train model on uncorrupted data
        lowest_val_loss = model.start_train(
            train_dl,
            valid_dl,
            device,
            checkpoint_dir=ckpt_dir,
            trial=trial,
            **train_params,
        )

        print("done backbone training")

        model.deactivate_uncert_head()
        valid_acc_backbone = model.eval_acc(valid_dl, device)
        mlflow.log_metric("valid_acc_backbone", valid_acc_backbone)
        print("backbone valid acc on uncorrupted: ", valid_acc_backbone)

        model.activate_uncert_head()

        # now train SPN
        train_params["num_epochs"] = backup_epochs
        train_params["warmup_epochs"] = 0

        # compute mean and std of corrupted data, s.t. we can use it to normalize it for SPN training
        model.compute_normalization_values(train_dl, device)

        print("training SPN")
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

        print("start evaluation")
        model.activate_uncert_head()
        # from mspn import MSPN_utils

        # mspn_util = MSPN_utils(model, spn, [0, 1, 2])

        # valid_data = mspn_util.create_data(valid_ds, valid_dl, device)
        # valid_corrupt_acc = mspn_util.eval_acc(valid_data)
        valid_corrupt_acc = model.eval_acc(valid_dl, device)
        mlflow.log_metric("valid_acc", valid_corrupt_acc)

        # Evaluate
        model.eval()
        train_ll_marg = model.eval_ll_marg(
            None,
            device,
            train_dl,
        )
        # train_data = mspn_util.create_data(train_ds, train_dl, device)
        # train_ll_marg = mspn_util.eval_ll_marg(train_data)
        mlflow.log_metric("train_ll_marg", train_ll_marg)

        valid_ll_marg = model.eval_ll_marg(
            None,
            device,
            valid_dl,
        )
        # valid_ll_marg = mspn_util.eval_ll_marg(valid_data)
        mlflow.log_metric("valid_ll_marg", valid_ll_marg)

        # Eval LL explanations
        # Level 5 gauss
        levels = [4]
        corruption = ["gaussian_noise"]
        print("eval level 5 corrupted")
        (
            test_corrupt_data,
            test_corrupt_levels,
        ) = get_corrupted_svhn_test(corruptions, corruption, levels)
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
        test_dl = DataLoader(
            list(
                zip(test_corrupt_data, test_ds.labels[int(len(test_ds.labels) / 2) :])
            ),
            batch_size=batch_sizes["resnet"],
            shuffle=False,
            pin_memory=True,
            num_workers=2,
        )
        # Should be low compared to training data
        level5_gauss_ll_marg = model.eval_ll_marg(None, device, test_dl)
        mlflow.log_metric("level5_gauss_ll_marg", level5_gauss_ll_marg)
        # explain LL
        # Should be high for the gaussian noise variable, low for the others
        level5_gauss_expl_ll = model.explain_ll(test_dl, device, return_all=False)
        level5_gauss_expl_ll = dict(
            zip(
                ["level5_gauss_expl_ll_" + c for c in corruptions], level5_gauss_expl_ll
            )
        )
        mlflow.log_metrics(level5_gauss_expl_ll)

        # Level 5 frost
        levels = [4]
        corruption = ["frost"]
        print("eval level 5 corrupted")
        (
            test_corrupt_data,
            test_corrupt_levels,
        ) = get_corrupted_svhn_test(corruptions, corruption, levels)
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
        test_dl = DataLoader(
            list(
                zip(test_corrupt_data, test_ds.labels[int(len(test_ds.labels) / 2) :])
            ),
            batch_size=batch_sizes["resnet"],
            shuffle=False,
            pin_memory=True,
            num_workers=2,
        )
        # Should be low compared to training data
        level5_frost_ll_marg = model.eval_ll_marg(None, device, test_dl)
        mlflow.log_metric("level5_frost_ll_marg", level5_frost_ll_marg)
        # explain LL
        # Should be high for the frost variable, low for the others
        level5_frost_expl_ll = model.explain_ll(test_dl, device, return_all=False)
        level5_frost_expl_ll = dict(
            zip(
                ["level5_frost_expl_ll_" + c for c in corruptions],
                level5_frost_expl_ll,
            )
        )
        mlflow.log_metrics(level5_frost_expl_ll)

        # Level 5 motion
        levels = [4]
        corruption = ["motion_blur"]
        print("eval level 5 corrupted")
        (
            test_corrupt_data,
            test_corrupt_levels,
        ) = get_corrupted_svhn_test(corruptions, corruption, levels)
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
        test_dl = DataLoader(
            list(
                zip(test_corrupt_data, test_ds.labels[int(len(test_ds.labels) / 2) :])
            ),
            batch_size=batch_sizes["resnet"],
            shuffle=False,
            pin_memory=True,
            num_workers=2,
        )
        # Should be low compared to training data
        level5_motion_ll_marg = model.eval_ll_marg(None, device, test_dl)
        mlflow.log_metric("level5_motion_ll_marg", level5_motion_ll_marg)
        # explain LL
        # Should be high for the motion variable, low for the others
        level5_motion_expl_ll = model.explain_ll(test_dl, device, return_all=False)
        level5_motion_expl_ll = dict(
            zip(
                ["level5_motion_expl_ll_" + c for c in corruptions],
                level5_motion_expl_ll,
            )
        )
        mlflow.log_metrics(level5_motion_expl_ll)

        # Eval MPE explanations
        # Level 1 gauss
        levels = [0]
        corruption = ["gaussian_noise"]
        print("eval level 1 corrupted")
        (
            test_corrupt_data,
            test_corrupt_levels,
        ) = get_corrupted_svhn_test(corruptions, corruption, levels)
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
        test_dl = DataLoader(
            list(
                zip(test_corrupt_data, test_ds.labels[int(len(test_ds.labels) / 2) :])
            ),
            batch_size=batch_sizes["resnet"],
            shuffle=False,
            pin_memory=True,
            num_workers=2,
        )
        # Should be within normal range (similar to valid)
        level1_gauss_ll_marg = model.eval_ll_marg(None, device, test_dl)
        mlflow.log_metric("level1_gauss_ll_marg", level1_gauss_ll_marg)

        # Should reflect the actual value of the explaining variables
        level1_gauss_mpe = model.explain_mpe(test_dl, device, return_all=False)
        level1_gauss_mpe = dict(
            zip(
                ["level1_gauss_mpe_" + c for c in corruptions],
                level1_gauss_mpe.tolist(),
            )
        )
        mlflow.log_metrics(level1_gauss_mpe)

        # Level 1 frost
        levels = [0]
        corruption = ["frost"]
        print("eval level 1 corrupted")
        (
            test_corrupt_data,
            test_corrupt_levels,
        ) = get_corrupted_svhn_test(corruptions, corruption, levels)
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
        test_dl = DataLoader(
            list(
                zip(test_corrupt_data, test_ds.labels[int(len(test_ds.labels) / 2) :])
            ),
            batch_size=batch_sizes["resnet"],
            shuffle=False,
            pin_memory=True,
            num_workers=2,
        )
        # Should be within normal range (similar to valid)
        level1_frost_ll_marg = model.eval_ll_marg(None, device, test_dl)
        mlflow.log_metric("level1_frost_ll_marg", level1_frost_ll_marg)

        # Should reflect the actual value of the explaining variables
        level1_frost_mpe = model.explain_mpe(test_dl, device, return_all=False)
        level1_frost_mpe = dict(
            zip(
                ["level1_frost_mpe_" + c for c in corruptions],
                level1_frost_mpe.tolist(),
            )
        )
        mlflow.log_metrics(level1_frost_mpe)

        # Level 1 motion
        levels = [0]
        corruption = ["motion_blur"]
        print("eval level 1 corrupted")
        (
            test_corrupt_data,
            test_corrupt_levels,
        ) = get_corrupted_svhn_test(corruptions, corruption, levels)
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
        test_dl = DataLoader(
            list(
                zip(test_corrupt_data, test_ds.labels[int(len(test_ds.labels) / 2) :])
            ),
            batch_size=batch_sizes["resnet"],
            shuffle=False,
            pin_memory=True,
            num_workers=2,
        )
        # Should be within normal range (similar to valid)
        level1_motion_ll_marg = model.eval_ll_marg(None, device, test_dl)
        mlflow.log_metric("level1_motion_ll_marg", level1_motion_ll_marg)

        # Should reflect the actual value of the explaining variables
        level1_motion_mpe = model.explain_mpe(test_dl, device, return_all=False)
        level1_motion_mpe = dict(
            zip(
                ["level1_motion_mpe_" + c for c in corruptions],
                level1_motion_mpe.tolist(),
            )
        )
        mlflow.log_metrics(level1_motion_mpe)
