import os
import numpy as np
from torch.utils.data import DataLoader
import torch
import mlflow


dataset_dir = "/data_docker/datasets/"


def load_datasets():
    # get normal cifar-10
    from torchvision.datasets import CIFAR10
    from torchvision import transforms

    # def add_brightness(imgs, distr="normal", value=None):
    #     # add or subtract brightness of all images
    #     # assumes imgs.shape = [batch_size, 32, 32, 3]
    #     # assumes imgs are in range [0, 255]
    #     # values are drawn from a normal distribution
    #     imgs_torch = torch.tensor(imgs, dtype=torch.float32)
    #     if value is None:
    #         if distr == "normal":
    #             values = torch.normal(0, 50, size=(imgs.shape[0],))
    #         elif distr == "uniform":
    #             values = torch.randint(-300, 300, size=(imgs.shape[0],))
    #         else:
    #             raise NotImplementedError
    #         values = values.reshape(-1, 1, 1, 1)

    #     else:
    #         # values = torch.normal(value, 20, size=(imgs.shape[0],))
    #         values = torch.ones((imgs.shape[0],)) * value
    #         values = values.reshape(-1, 1, 1, 1)
    #     imgs = torch.clamp(imgs_torch + values, 0, 255)
    #     imgs = imgs.to(dtype=torch.uint8).numpy()
    # return imgs, values

    def add_noise(imgs, value=None):
        imgs_torch = torch.tensor(imgs, dtype=torch.float32)
        if value is None:
            std_dev = torch.randint(0, 100, size=(imgs_torch.shape[0],))
        else:
            std_dev = torch.ones((imgs_torch.shape[0],)) * value

        noise = torch.randn(imgs_torch.shape) * std_dev.reshape(-1, 1, 1, 1)
        # for img in imgs_torch:
        #     random_choice = np.random.choice(100) / 10 #std dev of noise
        #     noise = torch.randn(img.shape) * random_choice #N(0,r)
        #     img + noise
        imgs = torch.clamp(imgs_torch + noise, 0, 255)
        imgs = imgs.to(dtype=torch.uint8).numpy()
        return imgs, std_dev

    test_transformer = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    train_ds = CIFAR10(root=dataset_dir + "cifar10", download=True, train=True)
    test_ds = CIFAR10(root=dataset_dir + "cifar10", download=True, train=False)

    # corrupt train-data with brightness (Normal-dist)
    # maximum +5 and -5
    # print first train image
    # train_ds_bright, values = add_brightness(train_ds.data.copy())
    train_ds_bright, values = add_noise(train_ds.data.copy())

    # find index in value where value > 100
    first_large = values > 98
    first_large = np.where(first_large)[0][0]
    first_small = values == 50
    first_small = np.where(first_small)[0][0]

    import matplotlib.pyplot as plt

    plt.imshow(train_ds_bright[first_large])
    # add value as title
    plt.title(f"brightness: {values[first_large]}")
    plt.savefig(f"low_blur_{first_large}.pdf")
    plt.clf()

    plt.imshow(train_ds.data[first_large])
    # add value as title
    plt.savefig(f"default_{first_large}.pdf")
    plt.clf()

    plt.imshow(train_ds_bright[first_small])
    # add value as title
    plt.title(f"brightness: {values[first_small]}")
    plt.savefig(f"high_blur_{first_small}.pdf")
    plt.clf()

    plt.imshow(train_ds.data[first_small])
    # add value as title
    plt.savefig(f"default_{first_small}.pdf")
    plt.clf()

    train_data = [test_transformer(img).flatten() for img in train_ds_bright]
    train_data = torch.concat(
        [
            values.reshape(-1, 1),
            torch.stack(train_data, dim=0),
        ],
        dim=1,
    )
    train_data = list(zip(train_data, train_ds.targets))

    # TODO: test_set should contain uniformly distributed brightnesses between -200 and 200
    # final plot should show how accuracy, Expl_l, Expl_p and MPE react differently
    # to the brightness values

    # create three test-sets: normal, brightness +100, brightness -100
    test_normal = [test_transformer(img).flatten() for img in test_ds.data]
    test_normal_data = torch.concat(
        [
            torch.zeros((len(test_normal), 1)),
            torch.stack(test_normal, dim=0),
        ],
        dim=1,
    )
    test_normal_ds = list(zip(test_normal_data, test_ds.targets))

    # test_uniform, values_u = add_brightness(test_ds.data.copy(), distr="uniform")
    test_uniform, values_u = add_noise(test_ds.data.copy())
    test_uniform = [test_transformer(img).flatten() for img in test_uniform]
    test_uniform_data = torch.concat(
        [
            values_u.reshape(-1, 1),
            torch.stack(test_uniform, dim=0),
        ],
        dim=1,
    )
    test_uniform_ds = list(zip(test_uniform_data, test_ds.targets))

    test_bright, values_b = add_noise(test_ds.data.copy(), value=90)
    test_bright = [test_transformer(img).flatten() for img in test_bright]
    test_bright_data = torch.concat(
        [
            values_b.reshape(-1, 1),
            torch.stack(test_bright, dim=0),
        ],
        dim=1,
    )
    test_bright_ds = list(zip(test_bright_data, test_ds.targets))

    test_dark, values_d = add_noise(test_ds.data.copy(), value=10)
    test_dark = [test_transformer(img).flatten() for img in test_dark]
    test_dark_data = torch.concat(
        [
            values_d.reshape(-1, 1),
            torch.stack(test_dark, dim=0),
        ],
        dim=1,
    )
    test_dark_ds = list(zip(test_dark_data, test_ds.targets))

    train_orig_ds = [test_transformer(img).flatten() for img in train_ds.data]
    train_orig_ds = torch.concat(
        [
            torch.zeros((len(train_orig_ds), 1)),
            torch.stack(train_orig_ds, dim=0),
        ],
        dim=1,
    )
    train_orig_ds = list(zip(train_orig_ds, train_ds.targets))

    # qualitative
    test_q_data = test_ds.data[:10]
    # levels = [-200, -150, -100, -50, 0, 50, 100, 150, 200]
    levels = range(0, 100, 20)
    test_q_ds = []
    test_q_images = []
    for l in levels:
        adjusted, _ = add_noise(test_q_data.copy(), value=l)
        test_q_images.append(adjusted)
        test_q = [test_transformer(img).flatten() for img in adjusted]
        test_q = torch.concat(
            [
                torch.ones((len(test_q), 1)) * l,
                torch.stack(test_q, dim=0),
            ],
            dim=1,
        )
        test_q = list(zip(test_q, test_ds.targets[:10]))
        test_q_ds.extend(test_q)

    return (
        train_orig_ds,
        train_data,
        test_normal_ds,
        test_uniform_ds,
        test_bright_ds,
        test_dark_ds,
        test_q_ds,
        test_q_images,
    )


def start_cifar10_brightness_run(
    run_name, batch_sizes, model_params, train_params, trial
):
    run_name += "_test"
    with mlflow.start_run(run_name=run_name) as run:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlflow.log_param("device", device)

        ckpt_dir = f"/data_docker/ckpts/cifar10-c_expl/{run_name}/"
        mlflow.log_param("ckpt_dir", ckpt_dir)

        # log all params
        mlflow.log_params(batch_sizes)
        mlflow.log_params(model_params)
        mlflow.log_params(train_params)

        corruption_levels_train = train_params["corruption_levels_train"]
        del train_params["corruption_levels_train"]

        # load data
        (
            train_orig_ds,
            train_ds_b,
            test_normal_ds,
            test_uniform_ds,
            test_bright_ds,
            test_dark_ds,
            test_q_ds,
            test_q_bright,
        ) = load_datasets()

        train_ds, valid_ds = torch.utils.data.random_split(
            train_orig_ds, [0.8, 0.2], generator=torch.Generator().manual_seed(0)
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

        train_ds_b, valid_ds_b = torch.utils.data.random_split(
            train_ds_b, [0.8, 0.2], generator=torch.Generator().manual_seed(0)
        )
        train_dl_b = DataLoader(
            train_ds_b,
            batch_size=batch_sizes["resnet"],
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )
        valid_dl_b = DataLoader(
            valid_ds_b,
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

        train_backbone_default = train_params["train_backbone_default"]
        del train_params["train_backbone_default"]

        print("training backbone")
        # train backbone model
        backup_epochs = train_params["num_epochs"]
        train_params["num_epochs"] = 0
        lowest_val_loss = model.start_train(
            train_dl if train_backbone_default else train_dl_b,
            valid_dl if train_backbone_default else valid_dl_b,
            device,
            checkpoint_dir=ckpt_dir,
            trial=trial,
            **train_params,
        )

        model.compute_normalization_values(train_dl, device)

        train_params["warmup_epochs"] = 0
        train_params["num_epochs"] = backup_epochs
        print("training einet")
        # train einet model
        lowest_val_loss_b = model.start_train(
            train_dl_b,
            valid_dl_b,
            device,
            checkpoint_dir=ckpt_dir,
            trial=trial,
            **train_params,
        )

        model.activate_uncert_head()
        # before costly evaluation, make sure that the model is not completely off
        valid_acc = model.eval_acc(valid_dl_b, device)
        mlflow.log_metric("valid_acc", valid_acc)
        model.deactivate_uncert_head()
        valid_acc = model.eval_acc(valid_dl_b, device)
        mlflow.log_metric("backbone_valid_acc", valid_acc)
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
        model.eval()

        model.embedding_histogram(valid_dl_b, device)

        # plot mpe histogram train
        mpe_b = model.explain_mpe(train_dl_b, device, return_all=True)
        mpe_b = torch.cat(mpe_b, dim=0)  # shape: [n_samples, 1]
        import matplotlib.pyplot as plt

        plt.hist(mpe_b.cpu().numpy(), bins=50)
        mlflow.log_figure(plt.gcf(), "train_mpe_hist.pdf")
        plt.clf()

        # all explanations on test_uniform_ds
        test_uniform_dl = DataLoader(
            test_uniform_ds,
            batch_size=batch_sizes["resnet"],
            shuffle=False,
            pin_memory=True,
            num_workers=2,
        )
        model.eval_acc(test_uniform_dl, device)
        # this should have the same order as the actual data
        # so i can use the first value of each data-point to get the actual brightness value (x-axis)
        # and the explanation can be on the y axis
        ll_expl = model.explain_ll(test_uniform_dl, device, True)
        p_expl = model.explain_posterior(test_uniform_dl, device, True)
        mpe_expl = model.explain_mpe(test_uniform_dl, device, True)

        # bin the x-axis values. They go from -200 to 200, so with 20 bins, each bin is 20 wide
        # bins = np.linspace(-300, 300, 21)
        bins = np.linspace(0, 100, 21)
        mpe_expl = torch.cat(mpe_expl, dim=0)
        brightness_vals = [
            test_uniform_ds[i][0][0].item() for i in range(len(test_uniform_ds))
        ]
        bright_bins = np.digitize(brightness_vals, bins)

        # compute acc of each bin
        accs = [
            model.eval_acc(
                DataLoader(
                    [
                        test_uniform_ds[i]
                        for i in range(len(test_uniform_ds))
                        if bright_bins[i] == j
                    ],
                    batch_size=batch_sizes["resnet"],
                    shuffle=False,
                    pin_memory=True,
                    num_workers=2,
                ),
                device,
            )
            for j in range(1, 21)
        ]

        ll_expl_binned = [
            ll_expl[bright_bins == i].mean().cpu().numpy() for i in range(1, 21)
        ]
        p_expl_binned = [
            p_expl[bright_bins == i].mean().cpu().numpy() for i in range(1, 21)
        ]
        mpe_expl_binned = [
            mpe_expl[bright_bins == i].mean().cpu().numpy() for i in range(1, 21)
        ]
        mlflow.log_metric("ll_min", np.min(ll_expl_binned))
        mlflow.log_metric("ll_max", np.max(ll_expl_binned))
        # normalize the explanations
        ll_expl_binned = (ll_expl_binned - np.min(ll_expl_binned)) / (
            np.max(ll_expl_binned) - np.min(ll_expl_binned)
        )
        mlflow.log_metric("p_min", np.min(p_expl_binned))
        mlflow.log_metric("p_max", np.max(p_expl_binned))
        p_expl_binned = (p_expl_binned - np.min(p_expl_binned)) / (
            np.max(p_expl_binned) - np.min(p_expl_binned)
        )
        mlflow.log_metric("mpe_min", np.min(mpe_expl_binned))
        mlflow.log_metric("mpe_max", np.max(mpe_expl_binned))
        mpe_expl_binned = (mpe_expl_binned - np.min(mpe_expl_binned)) / (
            np.max(mpe_expl_binned) - np.min(mpe_expl_binned)
        )

        from plotting_utils import plot_brightness_binned

        fig = plot_brightness_binned(
            bins, accs, ll_expl_binned, p_expl_binned, mpe_expl_binned
        )
        mlflow.log_figure(fig, "brightness_expl.pdf")

        # plot mpe histogram tests
        normal_dl = DataLoader(
            test_normal_ds,
            batch_size=batch_sizes["resnet"],
            shuffle=True,
            pin_memory=True,
            num_workers=1,
        )
        mpe_normal = model.explain_mpe(normal_dl, device, return_all=True)
        mpe_normal = torch.cat(mpe_normal, dim=0)
        plt.hist(mpe_normal.cpu().numpy(), bins=50)
        mlflow.log_figure(plt.gcf(), "normal_mpe_hist.pdf")
        plt.clf()

        bright_dl = DataLoader(
            test_bright_ds,
            batch_size=batch_sizes["resnet"],
            shuffle=True,
            pin_memory=True,
            num_workers=1,
        )
        mpe_bright = model.explain_mpe(bright_dl, device, return_all=True)
        mpe_bright = torch.cat(mpe_bright, dim=0)
        plt.hist(mpe_bright.cpu().numpy(), bins=50)
        mlflow.log_figure(plt.gcf(), "bright_mpe_hist.pdf")
        plt.clf()

        dark_dl = DataLoader(
            test_dark_ds,
            batch_size=batch_sizes["resnet"],
            shuffle=True,
            pin_memory=True,
            num_workers=1,
        )
        mpe_dark = model.explain_mpe(dark_dl, device, return_all=True)
        mpe_dark = torch.cat(mpe_dark, dim=0)
        plt.hist(mpe_dark.cpu().numpy(), bins=50)
        mlflow.log_figure(plt.gcf(), "dark_mpe_hist.pdf")
        plt.clf()

        # Idea: Use 10 test images, create 10 brightness levels of each, explain_mpe, show images and MPE
        test_q_dl = DataLoader(
            test_q_ds,
            batch_size=batch_sizes["resnet"],
            shuffle=False,
        )
        mpe_q = model.explain_mpe(test_q_dl, device, return_all=True)
        mpe_q = torch.cat(mpe_q, dim=0)
        for i in range(0, len(test_q_ds), 9):
            imgs = [test_q_bright[j][i // 9] for j in range(len(test_q_bright))]
            mpes = mpe_q[i : i + 9]
            fig, axes = plt.subplots(3, 3)
            axes_flat = axes.flatten()
            for ax, img, mpe in zip(axes_flat, imgs, mpes):
                ax.imshow(img)
                ax.axis("off")  # Turn off axis
                ax.set_title(mpe.item(), fontsize=10, pad=2)
            plt.tight_layout()
            mlflow.log_figure(fig, f"qualitative_{i}.pdf")
            plt.clf()
