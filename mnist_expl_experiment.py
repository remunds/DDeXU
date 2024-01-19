import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import os
import mlflow
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_datasets():
    data_dir = "/data_docker/datasets/"

    # load data
    mnist_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            # transforms.Lambda(lambda x: x.reshape(-1, 28 * 28).squeeze()),
        ]
    )

    # load mnist
    train_ds = datasets.MNIST(
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

    return train_ds, test_ds, ood_ds


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
        else 2
        if i < index_2
        else 3
        if i < index_3
        else 4
        if i < index_4
        else 5
    )


def get_manipulations(ds, highest_severity=None):
    if highest_severity:
        # choose randomly between 0 and highest_severity
        rotations = torch.randint(0, highest_severity + 1, (len(ds),))
        cutoffs = torch.randint(0, highest_severity + 1, (len(ds),))
        noises = torch.randint(0, highest_severity + 1, (len(ds),))
        return rotations, cutoffs, noises

    rotations = torch.zeros((len(ds),))
    cutoffs = torch.zeros((len(ds),))
    noises = torch.zeros((len(ds),))
    # 1: 20, 2: 40, 3: 60, 4: 80, 5: 100 degrees
    for i, r in enumerate(rotations):
        rotations[i] = get_severity(ds, i)

    # 1: 2, 2: 4, 3: 9, 4: 16, 5: 25 pixels
    for i, r in enumerate(cutoffs):
        cutoffs[i] = get_severity(ds, i)

    # 0: 10, 1: 20, 2: 30, 3: 40, 4: 50 noise
    for i, r in enumerate(noises):
        noises[i] = get_severity(ds, i)
    return (
        rotations.to(dtype=torch.int),
        cutoffs.to(dtype=torch.int),
        noises.to(dtype=torch.int),
    )


def manipulate_data(data, rotations, cutoffs, noises, seperated=False):
    # first rotate, then cutoff, then add noise
    if seperated:
        rot_ds = torch.zeros(len(data), 28 * 28)
        cut_ds = torch.zeros(len(data), 28 * 28)
        noise_ds = torch.zeros(len(data), 28 * 28)
    else:
        manip_ds = torch.zeros(len(data), 28 * 28)
    for i, img in enumerate(data):
        image = img.reshape(28, 28, 1)

        image_transformed = transforms.ToTensor()(image.numpy())
        image_rot = transforms.functional.rotate(
            img=image_transformed, angle=int(rotations[i] * 20)  # , fill=-mean / std
        )
        if seperated:
            rot_ds[i] = transforms.Normalize((0.1307,), (0.3081,))(image_rot).flatten()
            image_cutoff = image_transformed.clone()
        else:
            image_cutoff = image_rot

        # cutoff rows
        cutoff = cutoffs[i] * 5
        image_cutoff[:, :cutoff, :] = 0
        if seperated:
            cut_ds[i] = transforms.Normalize((0.1307,), (0.3081,))(
                image_cutoff
            ).flatten()
            image_noise = image_transformed.clone()
        else:
            image_noise = image_cutoff

        # scale image back to 0-255
        image_noise = image_noise * 255
        image_noise = image_noise.to(dtype=torch.uint8)
        this_noise = torch.randn((1, 28, 28)) * noises[i] * 10
        img_noise = torch.clamp(image_noise + this_noise, 0, 255)
        # scale back to 0-1
        img_noise = img_noise / 255

        if seperated:
            noise_ds[i] = transforms.Normalize((0.1307,), (0.3081,))(
                img_noise
            ).flatten()
        else:
            manip_ds[i] = transforms.Normalize((0.1307,), (0.3081,))(
                img_noise
            ).flatten()
    if seperated:
        return rot_ds, cut_ds, noise_ds
    return manip_ds


def manipulate_single_image(img, manipulation):
    rotation = manipulation[0]
    cutoff = manipulation[1]
    noise = manipulation[2]
    # first rotate, then cutoff, then add noise
    img_rot = transforms.ToTensor()(img.numpy())
    img_rot = transforms.functional.rotate(img=img_rot, angle=int(rotation * 20))
    img_cutoff = img_rot
    cutoff = cutoff * 5
    img_cutoff[:, :cutoff, :] = 0

    img_noise = img_cutoff * 255
    img_noise = img_noise.to(dtype=torch.uint8)
    this_noise = torch.randn((1, 28, 28)) * noise * 5
    img_noise = torch.clamp(img_noise + this_noise, 0, 255).to(dtype=torch.uint8)
    img_noise = img_noise / 255

    final_image: torch.Tensor = transforms.Normalize((0.1307,), (0.3081,))(img_noise)
    return final_image


def mnist_expl_manual_evaluation(model_params, path, device):
    run_name = f"mnist_expl_manual_evaluation_{model_params['model']}"
    print("starting run: ", run_name)
    with mlflow.start_run(run_name=run_name):
        # load model
        model_name = model_params["model"]
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

        model.load(path)
        model.activate_einet()
        model.to(device)

        # get first image of test-set
        _, test_ds, _ = get_datasets()
        img = test_ds.data[0]

        # perform some manipulations:
        # 1. only rotate with severity 4
        expl1 = torch.tensor([0, 4, 0])
        img1 = manipulate_single_image(img, expl1)
        img1 = torch.concatenate([expl1, img1.flatten()])

        expl2 = torch.tensor([0, 0, 4])
        img2 = manipulate_single_image(img, expl2)
        img2 = torch.concatenate([expl2, img2.flatten()])

        expl3 = torch.tensor([0, 4, 4])
        img3 = manipulate_single_image(img, expl3)
        img3 = torch.concatenate([expl3, img3.flatten()])

        expl4 = torch.tensor([4, 0, 0])
        img4 = manipulate_single_image(img, expl4)
        img4 = torch.concatenate([expl4, img4.flatten()])
        imgs = torch.stack([img1, img2, img3, img4])

        # stupid test
        # img4_1 = torch.concatenate([expl1, img4.flatten()])
        # img4_2 = torch.concatenate([expl2, img4.flatten()])
        # img4_3 = torch.concatenate([expl3, img4.flatten()])
        # imgs = torch.stack([img4_1, img4_2, img4_3, img4_4])

        # create dataloader
        target = test_ds.targets[0]
        dl = DataLoader(
            TensorDataset(
                imgs,
                torch.tensor([target, target, target, target]),
            ),
            batch_size=1,
        )
        # get LL and explanations of all images
        ll = model.eval_ll(dl, device, True)
        print("lls: ", ll)
        # expl_ll = model.explain_ll(dl, device, True)
        expl_ll = model.explain_ll(dl, device, True)
        print("expl_lls: ", expl_ll)
        var_vals, mpe_vals = model.explain_mpe(dl, device, True)
        print("var: ", var_vals)
        print("mpe: ", mpe_vals)

        for i, img in enumerate(imgs):
            img = img[3:]
            # plot image, title contains explanations
            fig, ax = plt.subplots()
            ax.imshow(img.reshape(28, 28), cmap="gray")
            ax.set_title(
                f"rot: {expl_ll[i][0]}, cut: {expl_ll[i][1]}, noise: {expl_ll[i][2]}"
            )
            mlflow.log_figure(fig, f"img_{i}.png")
            plt.close()


def start_mnist_expl_run(run_name, batch_sizes, model_params, train_params, trial):
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
        mnist_ds, test_ds, ood_ds = get_datasets()

        # for train, manipulate all randomly until highest severity
        # we can also have multiple manipulations per image
        highest_severity = train_params["highest_severity_train"]
        del train_params["highest_severity_train"]
        rot, cut, nois = get_manipulations(mnist_ds, highest_severity=highest_severity)
        train_ds = manipulate_data(mnist_ds.data, rot, cut, nois, seperated=False)

        # # plot original
        # import matplotlib.pyplot as plt

        # image = mnist_ds.data[0].reshape(28, 28)
        # fig, ax = plt.subplots()
        # ax.imshow(image, cmap="gray")
        # ax.set_title(f"label: {mnist_ds.targets[0]}")
        # mlflow.log_figure(fig, "original.png")

        # # show first image

        # image = train_ds[0].reshape(28, 28)
        # fig, ax = plt.subplots()
        # ax.imshow(image, cmap="gray")
        # ax.set_title(
        #     f"label: {mnist_ds.targets[0]}, rot: {rot[0]}, cut: {cut[0]}, noise: {nois[0]}"
        # )
        # mlflow.log_figure(fig, "manipulated.png")

        manipulations = torch.stack([rot, cut, nois], dim=1)
        # add rot, cut, noise in front of each image in train_ds
        train_ds = torch.cat([manipulations, train_ds], dim=1)
        train_ds = TensorDataset(train_ds, mnist_ds.targets)

        train_ds, valid_ds = torch.utils.data.random_split(
            train_ds, [0.8, 0.2], generator=torch.Generator().manual_seed(0)
        )

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

        # plot first 5 images
        for i in range(5):
            fig, ax = plt.subplots()
            ax.imshow(train_ds[i][0][3:].reshape(28, 28), cmap="gray")
            ax.set_title(
                f"label: {train_ds[i][1]}, rot: {train_ds[i][0][0]}, cut: {train_ds[i][0][1]}, noise: {train_ds[i][0][2]}"
            )
            mlflow.log_figure(fig, f"manipulated_{i}.png")

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

        if train_params["num_epochs"] == 0:
            return lowest_val_loss
        # evaluate
        model.eval()

        # eval accuracy
        model.einet_active = False
        valid_acc = model.eval_acc(valid_dl, device)
        mlflow.log_metric("valid_acc_resnet", valid_acc)
        model.einet_active = True
        valid_acc = model.eval_acc(valid_dl, device)
        mlflow.log_metric("valid_acc", valid_acc)

        valid_ll = model.eval_ll(valid_dl, device)
        mlflow.log_metric("valid_ll", valid_ll)

        del train_dl
        del valid_dl

        # for test, manipulate all separately
        rot, cut, nois = get_manipulations(test_ds)
        rot_ds, cut_ds, noise_ds = manipulate_data(
            test_ds.data, rot, cut, nois, seperated=True
        )
        rot = rot.unsqueeze(1)
        # add rot, cut, noise in front of each image in each test_ds
        rot_ds = torch.cat(
            [rot, torch.zeros(len(rot_ds), 1), torch.zeros(len(rot_ds), 1), rot_ds],
            dim=1,
        )
        cut = cut.unsqueeze(1)
        cut_ds = torch.cat(
            [torch.zeros(len(cut_ds), 1), cut, torch.zeros(len(cut_ds), 1), cut_ds],
            dim=1,
        )
        nois = nois.unsqueeze(1)
        noise_ds = torch.cat(
            [
                torch.zeros(len(noise_ds), 1),
                torch.zeros(len(noise_ds), 1),
                nois,
                noise_ds,
            ],
            dim=1,
        )

        # calibration test
        eval_dict = {}
        severity_levels = 5

        # if we have 5k images, 1k per severity level
        index_1 = len(test_ds) // severity_levels
        index_2 = index_1 * 2
        index_3 = index_1 * 3
        index_4 = index_1 * 4
        index_5 = index_1 * 5
        severity_indices = [index_1, index_2, index_3, index_4, index_5]
        prev_s = 0
        for s in tqdm(severity_indices):
            test_ds_rot_s = TensorDataset(rot_ds[prev_s:s], test_ds.targets[prev_s:s])
            test_dl_rot = DataLoader(
                test_ds_rot_s, batch_size=batch_sizes["resnet"], shuffle=True
            )
            severity = get_severity(test_ds, s - 1)
            acc = model.eval_acc(test_dl_rot, device)
            ll = model.eval_ll(test_dl_rot, device)
            expl_ll = model.explain_ll(test_dl_rot, device)
            expl_mpe = model.explain_mpe(test_dl_rot, device)
            if "rotation" not in eval_dict:
                eval_dict["rotation"] = {}
            eval_dict["rotation"][severity] = {
                "acc": acc,
                "ll": ll,
                # "expl_ll": expl_ll[0],
                "expl_ll": expl_ll,
                "expl_mpe": expl_mpe[0].item(),
            }
            # plot first image
            image = test_ds_rot_s[0][0][3:].reshape(28, 28)
            fig, ax = plt.subplots()
            ax.imshow(image, cmap="gray")
            ax.set_title(
                f"label: {test_ds_rot_s[0][1]}, rot: {test_ds_rot_s[0][0][0]}, cut: {test_ds_rot_s[0][0][1]}, noise: {test_ds_rot_s[0][0][2]}"
            )
            mlflow.log_figure(fig, f"rotation_{severity}.png")
            plt.close()

            test_ds_cutoff_s = TensorDataset(
                cut_ds[prev_s:s], test_ds.targets[prev_s:s]
            )
            test_dl_cutoff = DataLoader(
                test_ds_cutoff_s, batch_size=batch_sizes["resnet"], shuffle=True
            )
            # plot first image
            image = test_ds_cutoff_s[0][0][3:].reshape(28, 28)
            fig, ax = plt.subplots()
            ax.imshow(image, cmap="gray")
            ax.set_title(
                f"label: {test_ds_cutoff_s[0][1]}, rot: {test_ds_cutoff_s[0][0][0]}, cut: {test_ds_cutoff_s[0][0][1]}, noise: {test_ds_cutoff_s[0][0][2]}"
            )
            mlflow.log_figure(fig, f"cutoff_{severity}.png")
            plt.close()

            acc = model.eval_acc(test_dl_cutoff, device)
            ll = model.eval_ll(test_dl_cutoff, device)
            expl_ll = model.explain_ll(test_dl_cutoff, device)
            expl_mpe = model.explain_mpe(test_dl_cutoff, device)
            if "cutoff" not in eval_dict:
                eval_dict["cutoff"] = {}
            eval_dict["cutoff"][severity] = {
                "acc": acc,
                "ll": ll,
                # "expl_ll": expl_ll[1],
                "expl_ll": expl_ll,
                "expl_mpe": expl_mpe[1].item(),
            }

            test_ds_noise_s = TensorDataset(
                noise_ds[prev_s:s], test_ds.targets[prev_s:s]
            )
            test_dl_noise = DataLoader(
                test_ds_noise_s, batch_size=batch_sizes["resnet"], shuffle=True
            )

            # plot first image
            image = test_ds_noise_s[0][0][3:].reshape(28, 28)
            fig, ax = plt.subplots()
            ax.imshow(image, cmap="gray")
            ax.set_title(
                f"label: {test_ds_noise_s[0][1]}, rot: {test_ds_noise_s[0][0][0]}, cut: {test_ds_noise_s[0][0][1]}, noise: {test_ds_noise_s[0][0][2]}"
            )
            mlflow.log_figure(fig, f"noise_{severity}.png")
            plt.close()

            acc = model.eval_acc(test_dl_noise, device)
            ll = model.eval_ll(test_dl_noise, device)
            expl_ll = model.explain_ll(test_dl_noise, device)
            expl_mpe = model.explain_mpe(test_dl_noise, device)
            if "noise" not in eval_dict:
                eval_dict["noise"] = {}
            eval_dict["noise"][severity] = {
                "acc": acc,
                "ll": ll,
                # "expl_ll": expl_ll[2],
                "expl_ll": expl_ll,
                "expl_mpe": expl_mpe[2].item(),
            }
            prev_s = s

        mlflow.log_dict(eval_dict, "eval_dict")

        overall_acc = np.mean(
            [
                eval_dict[m][severity]["acc"]
                for m in eval_dict
                for severity in eval_dict[m]
            ]
        )
        overall_ll = np.mean(
            [
                eval_dict[m][severity]["ll"]
                for m in eval_dict
                for severity in eval_dict[m]
            ]
        )
        mlflow.log_metric("manip acc", overall_acc)
        mlflow.log_metric("manip ll", overall_ll)

        # create plot for each corruption
        # x axis: severity
        # y axis: acc, ll, var, entropy

        # TODO: add all ll-expl curves of all variables
        # this shows that only one explanation makes sense here!

        for m in ["rotation", "cutoff", "noise"]:
            accs = [
                eval_dict[m][severity]["acc"]
                for severity in sorted(eval_dict[m].keys())
            ]
            lls = [
                eval_dict[m][severity]["ll"] for severity in sorted(eval_dict[m].keys())
            ]
            ll_expl = [
                eval_dict[m][severity]["expl_ll"]
                for severity in sorted(eval_dict[m].keys())
            ]
            mpe_expl = [
                eval_dict[m][severity]["expl_mpe"]
                for severity in sorted(eval_dict[m].keys())
            ]
            fig, ax = plt.subplots()

            ax.set_xlabel("severity")
            ax.set_xticks(np.array(list(range(5))) + 1)

            ax.plot(accs, label="acc", color="red")
            ax.set_ylabel("accuracy", color="red")
            ax.tick_params(axis="y", labelcolor="red")
            ax.set_ylim([0, 1])

            ax2 = ax.twinx()
            ax2.plot(lls, label="ll", color="blue")
            ax2.tick_params(axis="y", labelcolor="blue")
            # ax2.set_ylim([0, 1])

            ax3 = ax.twinx()
            # TODO: cannot access these elements since its a list of list not tensor :(
            ll_expl = torch.tensor(ll_expl)
            ax3.plot(ll_expl[:, 0], label="ll expl rot", color="darkgreen")
            ax3.plot(ll_expl[:, 1], label="ll expl cut", color="mediumseagreen")
            ax3.plot(ll_expl[:, 2], label="ll expl noise", color="lime")
            # ax3.set_ylabel("predictive variance", color="green")
            ax3.tick_params(axis="y", labelcolor="green")

            ax4 = ax.twinx()
            ax4.plot(mpe_expl, label="mpe expl", color="orange")
            # ax4.set_ylabel("predictive entropy", color="orange")
            ax4.tick_params(axis="y", labelcolor="orange")

            fig.tight_layout()
            fig.legend()
            mlflow.log_figure(fig, f"{m}.png")
            plt.close()

        return lowest_val_loss
