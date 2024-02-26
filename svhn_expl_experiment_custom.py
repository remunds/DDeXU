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
            torch.ones((train_ds.data.shape[0], expl_len)),
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
    expl_vars = torch.ones((len(data), expl_len))
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
    test_corrupt_levels = np.ones(
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
    run_name = run_name + f"_custom"
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
                **model_params,
            )
        elif model_name == "AutoEncoderSPN":
            from ResNetSPN import AutoEncoderSPN

            model = AutoEncoderSPN(
                **model_params,
            )
        elif model_name == "EfficientNetSPN":
            from ResNetSPN import EfficientNetSPN

            model = EfficientNetSPN(
                **model_params,
            )
        else:
            raise NotImplementedError
        mlflow.set_tag("model", model.__class__.__name__)
        model = model.to(device)

        gaussian_noise_train = get_gaussian_noise_svhn_train(
            3, train_ds_orig.labels, test_transformer
        )  # train on level 3
        print(
            "gaussian_noise_train.shape: ",
            len(gaussian_noise_train),
            gaussian_noise_train[0][0].shape,
        )
        print("train_data.shape: ", len(train_data), train_data[0][0].shape)

        # take only half to reduce training time
        train_data = (
            train_data[: len(train_data) // 2]
            + gaussian_noise_train[len(train_data) // 2 :]
        )
        print("combined train_data.shape: ", len(train_data), train_data[0][0].shape)

        train_ds, valid_ds = torch.utils.data.random_split(
            train_data, [0.8, 0.2], generator=torch.Generator().manual_seed(0)
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

        print("getting embeddings")
        embeddings = model.get_embeddings(train_dl, device)
        from spn.structure.Base import Context
        from spn.structure.StatisticalTypes import MetaType
        from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian

        from spn.structure.Base import Sum, Product

        from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up

        # I want to create a SPN with the following structure:
        # 32 children (Normal) for the embeddings
        # 3 children (Categorical) for the corruption levels
        # somehow combine them
        # TODO: make sure its valid (i.e. decomposable and smooth)
        # Decomposable: product node is decomposable, if scopes of children do not share any variables
        # Smooth: all children of a sum node have the same scope
        # TODO: might also just use learn_spn for the embeddings and
        # add the categorical nodes manually
        spn_embedding = embeddings.cpu().numpy()

        # scopes: 0:32 embeddings (without corruptions), 32:35 corruptions
        # each embedding-input is a Normal, each corruption-input is a Categorical
        embedding_dists = []
        for i in range(model.num_hidden):
            embedding_dists.append(Gaussian(mean=0, stdev=1, scope=i))

        corruption_dists = []
        for i in range(expl_len):
            corruption_dists.append(
                Categorical(p=[1 / 5 for _ in range(5)], scope=model.num_hidden + i)
            )

        # now build spn from the input distributions
        products = []
        products2 = []
        for i in range(0, len(embedding_dists), 8):
            products.append(Product(children=embedding_dists[i : i + 8]))
            products2.append(Product(children=embedding_dists[i : i + 8]))

        sums = []
        for p1, p2 in zip(products, products2):
            sums.append(Sum(weights=[0.5, 0.5], children=[p1, p2]))

        large_prod1 = Product(children=sums + corruption_dists)
        large_prod2 = Product(children=sums + corruption_dists)

        spn = Sum(weights=[0.5, 0.5], children=[large_prod1, large_prod2])

        # para_types = [Gaussian for _ in range(spn_embedding.shape[1])]
        # para_types[:3] = [Categorical for _ in range(3)]
        # ds_context = Context(parametric_types=para_types).add_domains(spn_embedding)

        # from spn.algorithms.LearningWrappers import learn_parametric
        # spn = learn_parametric(spn_embedding, ds_context, min_instances_slice=1000)
        # prod1 = Product(
        #     children=[
        #         Categorical(p=[1 / 5 for _ in range(5)], scope=i)
        #         for i in range(expl_len)
        #     ]
        # )
        # spn = Product(children=[spn, prod1])

        assign_ids(spn)
        rebuild_scopes_bottom_up(spn)
        print(spn)
        from spn.algorithms.Validity import is_valid

        print(is_valid(spn))
        exit(0)

        print("embeddings shape: ", embeddings.shape)
        meta_types = [MetaType.REAL for _ in range(embeddings.shape[1])]
        meta_types.append(MetaType.REAL)
        meta_types[0] = MetaType.DISCRETE  # class
        meta_types[1] = MetaType.DISCRETE  # expl. var gaussian noise
        meta_types[2] = MetaType.DISCRETE  # expl. var defocus blur
        meta_types[3] = MetaType.DISCRETE  # expl. var contrast

        ds_context = Context(meta_types=meta_types)
        targets = np.array([t[1] for t in train_ds])
        train_data = embeddings.cpu().detach().numpy()
        # add target on pos1
        train_data = np.concatenate([targets[:, None], train_data], axis=1)
        ds_context.add_domains(train_data)

        from spn.algorithms.LearningWrappers import learn_mspn

        print("learning mspn")
        # spn = learn_mspn(train_data, ds_context, min_instances_slice=20)
        spn = learn_mspn(train_data, ds_context, min_instances_slice=200)
        # Note: changed stuff in LearningWrapper (assert for min/max domain)
        print("done learning mspn")

        print("start evaluation")
        model.activate_uncert_head()
        from mspn import MSPN_utils

        mspn_util = MSPN_utils(model, spn, [1, 2, 3])

        valid_data = mspn_util.create_data(valid_ds, valid_dl, device)
        valid_corrupt_acc = mspn_util.eval_acc(valid_data)
        mlflow.log_metric("valid_acc", valid_corrupt_acc)

        # Evaluate
        model.eval()
        # train_ll_marg = model.eval_ll_marg(
        #     None,
        #     device,
        #     train_dl,
        # )
        train_data = mspn_util.create_data(train_ds, train_dl, device)
        train_ll_marg = mspn_util.eval_ll_marg(train_data)
        mlflow.log_metric("train_ll_marg", train_ll_marg)

        # valid_ll_marg = model.eval_ll_marg(
        #     None,
        #     device,
        #     valid_dl,
        # )
        valid_ll_marg = mspn_util.eval_ll_marg(valid_data)
        mlflow.log_metric("valid_ll_marg", valid_ll_marg)

        # eval uncorrupted
        # expect high marginal LL, low entropy
        # expect MPE explanations all to be 1
        # expect LL explanations to be low
        print("eval uncorrupted")
        test_data = [test_transformer(img).flatten() for img, _ in test_ds]
        test_data = torch.concat(
            [
                torch.ones((test_ds.data.shape[0], expl_len)),
                torch.stack(test_data, dim=0),
            ],
            dim=1,
        )
        test_data = list(zip(test_data, test_ds.labels))
        test_dl = DataLoader(
            test_data,
            batch_size=batch_sizes["resnet"],
            shuffle=False,
            pin_memory=True,
            num_workers=2,
        )
        # test_ll_marg = model.eval_ll_marg(None, device, test_dl)
        test_data = mspn_util.create_data(test_ds, test_dl, device)
        test_ll_marg = mspn_util.eval_ll_marg(test_data)
        mlflow.log_metric("test_ll_marg", test_ll_marg)
        # test_mpe = model.explain_mpe(test_dl, device, return_all=False)
        test_mpe = mspn_util.explain_mpe(test_data, return_all=False)
        test_mpe = dict(
            zip(["test_mpe_" + c for c in corruptions_eval], test_mpe.tolist())
        )
        mlflow.log_metrics(test_mpe)
        # test_expl_ll = model.explain_ll(test_dl, device, return_all=False)
        test_expl_ll = mspn_util.explain_ll(test_data)
        test_expl_ll = dict(
            zip(["test_expl_ll_" + c for c in corruptions_eval], test_expl_ll)
        )
        mlflow.log_metrics(test_expl_ll)

        # eval gaussian corrupted
        # expect high marginal LL, high entropy
        # expect MPE explanations all to be 1 except for the gaussian variable, which should be 3
        # expect LL explanations to be low
        gauss_corruption = ["gaussian_noise"]
        level = [3]
        print("eval gaussian corrupted")
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
        print("gaussian corrupt shape: ", test_corrupt_data_gaussian.shape)
        gauss_dl = DataLoader(
            list(zip(test_corrupt_data_gaussian, test_ds.labels)),
            batch_size=batch_sizes["resnet"],
            shuffle=False,
            pin_memory=True,
            num_workers=2,
        )
        gauss_data = mspn_util.create_data(test_ds, gauss_dl, device)
        # gauss_ll_marg = model.eval_ll_marg(None, device, gauss_dl)
        gauss_ll_marg = mspn_util.eval_ll_marg(gauss_data)
        mlflow.log_metric("gauss_ll_marg", gauss_ll_marg)
        # gauss_mpe = model.explain_mpe(gauss_dl, device, return_all=False)
        gauss_mpe = mspn_util.explain_mpe(gauss_data, return_all=False)
        gauss_mpe = dict(
            zip(["gauss_mpe_" + c for c in corruptions_eval], gauss_mpe.tolist())
        )
        mlflow.log_metrics(gauss_mpe)
        # gauss_expl_ll = model.explain_ll(gauss_dl, device, return_all=False)
        gauss_expl_ll = mspn_util.explain_ll(gauss_data)
        gauss_expl_ll = dict(
            zip(["gauss_expl_ll_" + c for c in corruptions_eval], gauss_expl_ll)
        )
        mlflow.log_metrics(gauss_expl_ll)

        # eval defocus blur corrupted
        # expect low marginal LL, any (probably high) entropy
        # expect MPE explanations all to be close to 1 (not meaningful)
        # expect LL explanation to be high for the defocus blur variable, low for the others
        defocus_corruption = ["defocus_blur"]
        level = [3]
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
        print("defocus corrupt shape: ", test_corrupt_data_defocus.shape)
        defocus_dl = DataLoader(
            list(zip(test_corrupt_data_defocus, test_ds.labels)),
            batch_size=batch_sizes["resnet"],
            shuffle=False,
            pin_memory=True,
            num_workers=2,
        )
        defocus_data = mspn_util.create_data(test_ds, defocus_dl, device)
        # defocus_ll_marg = model.eval_ll_marg(None, device, defocus_dl)
        defocus_ll_marg = mspn_util.eval_ll_marg(defocus_data)
        mlflow.log_metric("defocus_ll_marg", defocus_ll_marg)
        # defocus_mpe = model.explain_mpe(defocus_dl, device, return_all=False)
        defocus_mpe = mspn_util.explain_mpe(defocus_data, return_all=False)
        defocus_mpe = dict(
            zip(["defocus_mpe_" + c for c in corruptions_eval], defocus_mpe.tolist())
        )
        mlflow.log_metrics(defocus_mpe)
        # defocus_expl_ll = model.explain_ll(defocus_dl, device, return_all=False)
        defocus_expl_ll = mspn_util.explain_ll(defocus_data)
        defocus_expl_ll = dict(
            zip(["defocus_expl_ll_" + c for c in corruptions_eval], defocus_expl_ll)
        )
        mlflow.log_metrics(defocus_expl_ll)

        # eval contrast corrupted
        # expect low marginal LL, any (probably high) entropy
        # expect MPE explanations all to be close to 1 (not meaningful)
        # expect LL explanation to be high for the contrast variable, low for the others
        contrast_corruption = ["contrast"]
        level = [3]
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
        print("contrast corrupt shape: ", test_corrupt_data_contrast.shape)
        contrast_dl = DataLoader(
            list(zip(test_corrupt_data_contrast, test_ds.labels)),
            batch_size=batch_sizes["resnet"],
            shuffle=False,
            pin_memory=True,
            num_workers=2,
        )
        contrast_data = mspn_util.create_data(test_ds, contrast_dl, device)
        # contrast_ll_marg = model.eval_ll_marg(None, device, contrast_dl)
        contrast_ll_marg = mspn_util.eval_ll_marg(contrast_data)
        mlflow.log_metric("contrast_ll_marg", contrast_ll_marg)
        # contrast_mpe = model.explain_mpe(contrast_dl, device, return_all=False)
        contrast_mpe = mspn_util.explain_mpe(contrast_data, return_all=False)
        contrast_mpe = dict(
            zip(["contrast_mpe_" + c for c in corruptions_eval], contrast_mpe.tolist())
        )
        mlflow.log_metrics(contrast_mpe)
        # contrast_expl_ll = model.explain_ll(contrast_dl, device, return_all=False)
        contrast_expl_ll = mspn_util.explain_ll(contrast_data)
        contrast_expl_ll = dict(
            zip(["contrast_expl_ll_" + c for c in corruptions_eval], contrast_expl_ll)
        )
        mlflow.log_metrics(contrast_expl_ll)
