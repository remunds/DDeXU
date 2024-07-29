import os
import numpy as np
from torch.utils.data import DataLoader
import torch
import mlflow
from torchvision.datasets import SVHN
from PIL import Image


dataset_dir = "/data_docker/datasets/"
svhn_c_path = "svhn_c"
svhn_c_path_complete = dataset_dir + svhn_c_path
corruptions_train = [
    "gaussian_noise",
    # "brightness",
    # "contrast",
    # "defocus_blur",
    # "elastic_transform",
    # "fog",
    # "frost",
    # "gaussian_blur", # not available
    # "glass_blur",
    # "impulse_noise",
    # "jpeg_compression",
    # "motion_blur",
    # "pixelate",
    # "saturate", # not available
    # "shot_noise",
    # "snow",
    # "spatter", # not available
    # "speckle_noise", # not available
    # "zoom_blur",
]
corruptions_eval = [
    # "brightness",
    "gaussian_noise",
    "contrast",
    "defocus_blur",
    # "elastic_transform",
    # "fog",
    # "frost",
    # "gaussian_blur", # not available
    # "glass_blur",
    # "impulse_noise",
    # "jpeg_compression",
    # "motion_blur",
    # "pixelate",
    # "saturate", # not available
    # "shot_noise",
    # "snow",
    # "spatter", # not available
    # "speckle_noise", # not available
    # "zoom_blur",
]
expl_len = len(corruptions_eval)

# This experiments should verify that the explanations can be used for the different kind of uncertainties
# MPE-explanations work with examples that are within the data distribution, but ambiguous -> Aleatoric uncertainty
# LL-explanations work with examples that are outside the data distribution -> Epistemic uncertainty
# We train the model on uncorrupted together with one corruption-type.
# For corruption-types it was trained on, the model should have low marginal LL uncertainty (epistemic), but high entropy (aleatoric).
# The expectation is that the model can explain these with MPE-explanations (given nothing).
# For corruption-types it was not trained on, the model should have high marginal LL uncertainty (epistemic).
# The expectation is that the model can explain these with LL-explanations (given the actual values).

# We train on SVHN, and Gaussian Noise of level3 on SVHN
# We evaluate on Gaussian Noise (aleatoric, MPE) and Defocus Blur, Contrast (epistemic, LL) on level 3.


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
            torch.zeros((train_ds.data.shape[0], expl_len)),
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


def get_gaussian_noise_svhn_train(level: int, train_labels: np.ndarray, transformer):
    data = np.load(f"{svhn_c_path_complete}/svhn_train_gaussian_noise_l{level}.npy")
    data = [transformer(img).flatten() for img in data]
    expl_vars = torch.zeros((len(data), expl_len))
    expl_vars[:, 0] = level
    # expl vars shape = N, 3
    # data shape = N, 32*32*3
    # result shape = N, 3+32*32*3
    data = torch.concat([expl_vars, torch.stack(data, dim=0)], dim=1)
    data = list(zip(data, train_labels))
    return data


def get_corrupted_svhn_train(all_corruption_len: int, corruptions: list, levels: list):
    # TODO change sample-size
    num_samples = 73257
    # available corrupted dataset: 26032*len(corruptions), 32, 32, 3
    datasets_length = num_samples * len(levels) * len(corruptions)
    train_corrupt_data = np.zeros((datasets_length, 32, 32, 3), dtype=np.uint8)
    train_corrupt_levels = np.zeros(
        (datasets_length, all_corruption_len), dtype=np.uint8
    )
    train_idx = 0
    for corr_idx, c in enumerate(corruptions):
        # each corrupted dataset has shape of test: 26032, 32, 32, 3
        for i in range(5):  # iterate over corruption levels
            if not i in levels:
                continue
            data = np.load(f"{svhn_c_path_complete}/svhn_train_{c}_l{i+1}.npy")

            new_train_idx = train_idx + num_samples
            train_corrupt_data[train_idx:new_train_idx] = data[:num_samples, ...]
            train_corrupt_levels[train_idx:new_train_idx, corr_idx] = i + 1

            train_idx = new_train_idx

    print("done loading train corruptions")
    return (
        train_corrupt_data,
        train_corrupt_levels,
    )


def get_corrupted_svhn_test(all_corruptions: list, corruptions: list, levels: list):
    num_samples = 26032
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
        for i in range(1, 5 + 1):  # iterate over corruption levels
            if not i in levels:
                continue
            data = np.load(f"{svhn_c_path_complete}/svhn_test_{c}_l{i}.npy")

            new_test_idx = test_idx + num_samples
            test_corrupt_data[test_idx:new_test_idx] = data[:num_samples, ...]
            test_corrupt_levels[test_idx:new_test_idx, corr_idx] = i

            test_idx = new_test_idx

    print("done loading test corruptions")
    return (
        test_corrupt_data,
        test_corrupt_levels,
    )


def start_svhn_expl_run(run_name, batch_sizes, model_params, train_params, trial):
    # run_name = run_name + f"_categorical_no_quant"
    run_name = run_name + f"_EM"
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

        gaussian_noise_train = get_gaussian_noise_svhn_train(
            1, train_ds_orig.labels, test_transformer
        )  # train on level 3
        print(
            "gaussian_noise_train.shape: ",
            len(gaussian_noise_train),
            gaussian_noise_train[0][0].shape,
        )
        print("train_data.shape: ", len(train_data), train_data[0][0].shape)

        # take only half to reduce training time
        # train_data = (
        #     train_data[: len(train_data) // 2]
        #     + gaussian_noise_train[len(train_data) // 2 :]
        # )
        # print("combined train_data.shape: ", len(train_data), train_data[0][0].shape)

        train_ds, valid_ds = torch.utils.data.random_split(
            train_data, [0.8, 0.2], generator=torch.Generator().manual_seed(0)
        )
        print("first 5, 5 in train: ")
        for i in range(5):
            print(train_ds[i][0][:5])

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

        # train model
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

        model.activate_uncert_head()

        # now train SPN
        train_params["num_epochs"] = backup_epochs
        train_params["warmup_epochs"] = 0

        # compute mean and std of corrupted data, s.t. we can use it to normalize it for SPN training
        model.compute_normalization_values(train_dl, device)

        # print("training SPN")
        # # train model
        # lowest_val_loss = model.start_train(
        #     train_dl,
        #     valid_dl,
        #     device,
        #     checkpoint_dir=ckpt_dir,
        #     trial=trial,
        #     **train_params,
        # )

        mlflow.pytorch.log_state_dict(model.state_dict(), "model")
        from EinsumNetwork import Graph, EinsumNetwork

        depth = 5
        num_repetitions = 10
        num_input_distributions = 15
        num_sums = 11

        max_num_epochs = 8
        batch_size = 512
        online_em_frequency = 1
        online_em_stepsize = 0.05

        # train_N, num_dims = train_x.shape
        # valid_N = valid_x.shape[0]
        # test_N = test_x.shape[0]
        num_dims = model.num_hidden + expl_len
        # graph = Graph.random_binary_trees(
        #     num_var=num_dims, depth=depth, num_repetitions=num_repetitions
        # )
        pd_num_pieces = [4]
        pd_delta = [[num_dims / d] for d in pd_num_pieces]
        graph = Graph.poon_domingos_structure(shape=(num_dims,), delta=pd_delta)

        args = EinsumNetwork.Args(
            num_classes=1,
            num_input_distributions=num_input_distributions,
            # exponential_family=EinsumNetwork.BinomialArray,
            # exponential_family_args={"N": 250},
            exponential_family=EinsumNetwork.NormalArray,
            num_sums=num_sums,
            num_var=num_dims,
            online_em_frequency=1,
            online_em_stepsize=0.05,
        )

        einet = EinsumNetwork.EinsumNetwork(graph, args)
        einet.initialize()
        einet.to(device)
        print(einet)

        # take only half to reduce training time
        train_data = (
            train_data[: len(train_data) // 2]
            + gaussian_noise_train[len(train_data) // 2 :]
        )
        print("combined train_data.shape: ", len(train_data), train_data[0][0].shape)

        train_ds, valid_ds = torch.utils.data.random_split(
            train_data, [0.8, 0.2], generator=torch.Generator().manual_seed(0)
        )
        print("first 5, 5 in combined-train: ")
        for i in range(5):
            print(train_ds[i][0][:5])
        gaussian_mean = 0
        for d, t in train_ds:
            gaussian_mean += d[0]
        gaussian_mean /= len(train_ds)
        print("train gaussian_mean: ", gaussian_mean)
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_sizes["resnet"],
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )

        print("getting embeddings")
        embeddings = model.get_embeddings(train_dl, device)
        embeddings = model.quantify(embeddings)
        train_x = embeddings
        print(train_x.shape)
        print(train_x[:10, :10])
        train_N = len(train_x)

        embeddings_val = model.get_embeddings(valid_dl, device)
        embeddings_val = model.quantify(embeddings_val)
        valid_x = embeddings_val
        valid_N = len(valid_x)

        print("starting training")
        for epoch_count in range(max_num_epochs):

            # evaluate
            train_ll = EinsumNetwork.eval_loglikelihood_batched(einet, train_x)
            valid_ll = EinsumNetwork.eval_loglikelihood_batched(einet, valid_x)
            # test_ll = EinsumNetwork.eval_loglikelihood_batched(einet, test_x)

            print(
                # "[{}]   train LL {}   valid LL {}  test LL {}".format(
                "[{}]   train LL {}   valid LL {}".format(
                    epoch_count,
                    train_ll / train_N,
                    valid_ll / valid_N,
                    # test_ll / test_N,
                )
            )

            # train
            idx_batches = torch.randperm(train_N).split(batch_size)
            for batch_count, idx in enumerate(idx_batches):
                batch_x = train_x[idx, :]
                outputs = einet.forward(batch_x)

                ll_sample = EinsumNetwork.log_likelihoods(outputs)
                log_likelihood = ll_sample.sum()

                objective = log_likelihood
                objective.backward()

                einet.em_process_batch()

            einet.em_update()

        print("done training")

        print("eval uncorrupted")
        test_data = [test_transformer(img).flatten() for img, _ in test_ds]
        test_data = torch.concat(
            [
                torch.zeros((test_ds.data.shape[0], expl_len)),
                torch.stack(test_data, dim=0),
            ],
            dim=1,
        )
        test_dl = DataLoader(
            test_data,
            batch_size=batch_sizes["resnet"],
            shuffle=True,
            pin_memory=True,
            num_workers=2,
        )
        test_data = model.get_embeddings(test_dl, device)
        test_data = model.quantify(test_data)

        test_ll = EinsumNetwork.eval_loglikelihood_batched(einet, test_data)
        print("uncorrupted_ll: ", test_ll / len(test_data))
        mlflow.log_metric("uncorrupted_ll", test_ll / len(test_data))
        marginalize_idxs = [0, 1, 2]
        einet.set_marginalization_idx(marginalize_idxs)
        test_data[:100, marginalize_idxs] = 0

        print("starting MPE")
        mpe_reconstruction = einet.mpe(x=test_data[:100]).cpu().numpy()
        mpe_reconstruction = mpe_reconstruction.squeeze()
        mpe_reconstruction = mpe_reconstruction.reshape((-1, 35))
        mpe_reconstruction = mpe_reconstruction[:, :3].mean(axis=0)
        test_mpe = dict(
            zip(
                ["test_mpe_" + c for c in corruptions_eval], mpe_reconstruction.tolist()
            )
        )
        print("test_mpe: ", test_mpe)
        mlflow.log_metrics(test_mpe)

        gauss_corruption = ["gaussian_noise"]
        level = [1]
        (
            test_corrupt_data_gaussian,
            test_corrupt_levels_gaussian,
        ) = get_corrupted_svhn_test(corruptions_eval, gauss_corruption, level)
        test_corrupt_data_gaussian = [
            test_transformer(img).flatten() for img in test_corrupt_data_gaussian
        ]
        test_corrupt_data_gaussian = torch.concat(
            [
                torch.from_numpy(test_corrupt_levels_gaussian).to(dtype=torch.int32),
                torch.stack(test_corrupt_data_gaussian, dim=0).to(dtype=torch.float32),
            ],
            dim=1,
        )
        print("first 5, 5 in gaussian: ", test_corrupt_data_gaussian[:5, :5])
        gauss_dl = DataLoader(
            test_corrupt_data_gaussian,
            batch_size=batch_sizes["resnet"],
            shuffle=True,
            pin_memory=True,
            num_workers=2,
        )
        test_corrupt_data_gaussian = model.get_embeddings(gauss_dl, device)
        test_corrupt_data_gaussian = model.quantify(test_corrupt_data_gaussian)

        print("gaussian corrupt shape: ", test_corrupt_data_gaussian.shape)
        gauss_ll = EinsumNetwork.eval_loglikelihood_batched(
            einet, test_corrupt_data_gaussian
        )
        print("gaussian_ll: ", gauss_ll / len(test_corrupt_data_gaussian))
        mlflow.log_metric("gaussian_ll", gauss_ll / len(test_corrupt_data_gaussian))
        print("starting MPE")
        einet.set_marginalization_idx(marginalize_idxs)
        test_corrupt_data_gaussian[:100, marginalize_idxs] = 0
        # choose 500 random samples
        random_samples = np.random.choice(
            test_corrupt_data_gaussian.shape[0], 500, replace=False
        )
        test_corrupt_data_gaussian = test_corrupt_data_gaussian[random_samples]
        real_gaussian = test_corrupt_data_gaussian[:, 0].mean(axis=0)
        print("real_gaussian: ", real_gaussian)
        mpe_reconstruction = einet.mpe(x=test_corrupt_data_gaussian).cpu().numpy()
        mpe_reconstruction = mpe_reconstruction.squeeze()
        mpe_reconstruction = mpe_reconstruction.reshape((-1, 35))
        print("mpe_reconstruction[:5, :3]: ", mpe_reconstruction[:5, :3])
        mpe_reconstruction = mpe_reconstruction[:, :3].mean(axis=0)
        gauss_mpe = dict(
            zip(
                ["gauss_mpe_" + c for c in corruptions_eval],
                mpe_reconstruction.tolist(),
            )
        )
        print("gauss_mpe: ", gauss_mpe)
        mlflow.log_metrics(gauss_mpe)

        defocus_corruption = ["defocus_blur"]
        level = [1]
        print("eval defocus corrupted")
        (
            test_corrupt_data_defocus,
            test_corrupt_levels_defocus,
        ) = get_corrupted_svhn_test(corruptions_eval, defocus_corruption, level)
        test_corrupt_data_defocus = [
            test_transformer(img).flatten() for img in test_corrupt_data_defocus
        ]
        test_corrupt_data_defocus = torch.concat(
            [
                torch.from_numpy(test_corrupt_levels_defocus).to(dtype=torch.int32),
                torch.stack(test_corrupt_data_defocus, dim=0).to(dtype=torch.float32),
            ],
            dim=1,
        )
        defocus_dl = DataLoader(
            test_corrupt_data_defocus,
            batch_size=batch_sizes["resnet"],
            shuffle=True,
            pin_memory=True,
            num_workers=2,
        )
        test_corrupt_data_defocus = model.get_embeddings(defocus_dl, device)
        test_corrupt_data_defocus = model.quantify(test_corrupt_data_defocus)
        print("defocus corrupt shape: ", test_corrupt_data_defocus.shape)
        defocus_corruption_ll = EinsumNetwork.eval_loglikelihood_batched(
            einet, test_corrupt_data_defocus
        )
        print(
            "defocus_corruption_ll: ",
            defocus_corruption_ll / len(test_corrupt_data_defocus),
        )
        mlflow.log_metric(
            "defocus_corruption_ll",
            defocus_corruption_ll / len(test_corrupt_data_defocus),
        )
        print("starting MPE")
        einet.set_marginalization_idx(marginalize_idxs)
        test_corrupt_data_defocus[:100, marginalize_idxs] = 0
        mpe_reconstruction = einet.mpe(x=test_corrupt_data_defocus[:100]).cpu().numpy()
        mpe_reconstruction = mpe_reconstruction.squeeze()
        mpe_reconstruction = mpe_reconstruction.reshape((-1, 35))
        print("mpe_reconstruction[:5, :3]: ", mpe_reconstruction[:5, :3])
        mpe_reconstruction = mpe_reconstruction[:, :3].mean(axis=0)
        defocus_mpe = dict(
            zip(
                ["defocus_mpe_" + c for c in corruptions_eval],
                mpe_reconstruction.tolist(),
            )
        )
        print("defocus_mpe: ", defocus_mpe)
        mlflow.log_metrics(defocus_mpe)

        contrast_corruption = ["contrast"]
        level = [1]
        print("eval contrast corrupted")
        (
            test_corrupt_data_contrast,
            test_corrupt_levels_contrast,
        ) = get_corrupted_svhn_test(corruptions_eval, contrast_corruption, level)
        test_corrupt_data_contrast = [
            test_transformer(img).flatten() for img in test_corrupt_data_contrast
        ]
        test_corrupt_data_contrast = torch.concat(
            [
                torch.from_numpy(test_corrupt_levels_contrast).to(dtype=torch.int32),
                torch.stack(test_corrupt_data_contrast, dim=0).to(dtype=torch.float32),
            ],
            dim=1,
        )
        contrast_dl = DataLoader(
            test_corrupt_data_contrast,
            batch_size=batch_sizes["resnet"],
            shuffle=True,
            pin_memory=True,
            num_workers=2,
        )
        test_corrupt_data_contrast = model.get_embeddings(contrast_dl, device)
        test_corrupt_data_contrast = model.quantify(test_corrupt_data_contrast)
        print("contrast corrupt shape: ", test_corrupt_data_contrast.shape)
        contrast_corruption_ll = EinsumNetwork.eval_loglikelihood_batched(
            einet, test_corrupt_data_contrast
        )
        print(
            "contrast_corruption_ll: ",
            contrast_corruption_ll / len(test_corrupt_data_contrast),
        )
        mlflow.log_metric(
            "contrast_corruption_ll",
            contrast_corruption_ll / len(test_corrupt_data_contrast),
        )
        print("starting MPE")
        einet.set_marginalization_idx(marginalize_idxs)
        test_corrupt_data_contrast[:100, marginalize_idxs] = 0
        mpe_reconstruction = einet.mpe(x=test_corrupt_data_contrast[:100]).cpu().numpy()
        mpe_reconstruction = mpe_reconstruction.squeeze()
        mpe_reconstruction = mpe_reconstruction.reshape((-1, 35))
        print("mpe_reconstruction[:5, :3]: ", mpe_reconstruction[:5, :3])
        mpe_reconstruction = mpe_reconstruction[:, :3].mean(axis=0)
        contrast_mpe = dict(
            zip(
                ["contrast_mpe_" + c for c in corruptions_eval],
                mpe_reconstruction.tolist(),
            )
        )
        print("contrast_mpe: ", contrast_mpe)
        mlflow.log_metrics(contrast_mpe)
