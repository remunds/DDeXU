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
        else 2 if i < index_2 else 3 if i < index_3 else 4 if i < index_4 else 5
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


def mnist_expl_qualitative_evaluation(model, device):
    model.to(device)
    model.activate_uncert_head()
    # get first image of test-set
    _, test_ds, _ = get_datasets()
    image = test_ds.data[0]

    imgs = []

    for i in range(0, 5, 2):
        for j in range(0, 5, 2):
            for k in range(0, 5, 2):
                # perform some manipulations:
                expl = torch.tensor([i, j, k])
                img = manipulate_single_image(image.clone(), expl)
                img = torch.concatenate([expl, img.flatten()])
                imgs.append(img)

    imgs = torch.stack(imgs)
    targets = test_ds.targets[0].repeat(len(imgs))

    # create dataloader
    dl = DataLoader(
        TensorDataset(
            imgs,
            targets,
        ),
        batch_size=1,
    )
    expl_ll = model.explain_ll(dl, device, True)
    expl_mpe = model.explain_mpe(dl, device, True)
    posteriors = model.eval_posterior(None, device, dl, True)
    preds = torch.max(posteriors, 1)[1]

    def create_image(explaining_vars, explanations, img, mode, pred):
        # plot image, title contains explanations
        fig, ax = plt.subplots()
        ax.imshow(img.reshape(28, 28), cmap="gray")
        ax.set_title(
            f"{pred}, rot: {explanations[0]:.3f}, cut: {explanations[1]:.3f}, noise: {explanations[2]:.3f}"
        )
        mlflow.log_figure(
            fig,
            f"img_{explaining_vars[0]}_{explaining_vars[1]}_{explaining_vars[2]}_{mode}.png",
        )
        plt.close()

    for i, img in enumerate(imgs):
        expl = img[:3]
        img = img[3:]
        pred = preds[i].item()
        create_image(expl, expl_ll[i], img, "ll", pred)
        create_image(expl, expl_mpe[i][0].tolist(), img, "mpe", pred)


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

        mnist_expl_qualitative_evaluation(model, device)


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
        elif model_name == "ConvResNetDDUGMM":
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
                **model_params,
            )
        elif model_name == "EfficientNetGMM":
            from ResNetSPN import EfficientNetGMM

            model = EfficientNetGMM(
                **model_params,
            )
        else:
            raise NotImplementedError
        mlflow.set_tag("model", model.__class__.__name__)
        model = model.to(device)

        # for train, manipulate all randomly until highest severity
        # we can also have multiple manipulations per image
        highest_severity = train_params["highest_severity_train"]
        del train_params["highest_severity_train"]
        rot, cut, nois = get_manipulations(mnist_ds, highest_severity=highest_severity)
        train_ds = manipulate_data(mnist_ds.data, rot, cut, nois, seperated=False)

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
            shuffle=False,
            pin_memory=True,
            num_workers=2,
        )
        valid_dl = DataLoader(
            valid_ds,
            batch_size=batch_sizes["resnet"],
            shuffle=False,
            pin_memory=True,
            num_workers=2,
        )
        print("train backbone on mnist")
        train_params["num_epochs"] = 0
        model.start_train(
            train_dl,
            valid_dl,
            device,
            checkpoint_dir=ckpt_dir,
            trial=trial,
            **train_params,
        )
        print("done train backbone")
        model.deactivate_uncert_head()
        valid_acc = model.eval_acc(valid_dl, device)
        mlflow.log_metric("valid_acc_backbone", valid_acc)
        model.activate_uncert_head()

        # MSPN
        model.eval()
        print("getting embeddings")
        embeddings = model.get_embeddings(train_dl, device)
        from spn.structure.Base import Context
        from spn.structure.StatisticalTypes import MetaType

        print("embeddings shape: ", embeddings.shape)
        meta_types = [MetaType.REAL for _ in range(embeddings.shape[1])]
        meta_types.append(MetaType.REAL)
        meta_types[0] = MetaType.DISCRETE  # class
        meta_types[1] = MetaType.DISCRETE  # expl. var
        meta_types[2] = MetaType.DISCRETE  # expl. var
        meta_types[3] = MetaType.DISCRETE  # expl. var

        ds_context = Context(meta_types=meta_types)
        targets = train_ds.tensors[1].cpu().detach().numpy()
        train_data = embeddings.cpu().detach().numpy()
        train_data = np.concatenate([targets[:, None], train_data], axis=1)
        ds_context.add_domains(train_data)

        from spn.algorithms.LearningWrappers import learn_mspn

        print("learning mspn")
        spn = learn_mspn(train_data, ds_context, min_instances_slice=20)
        print("done learning mspn")

        from spn.algorithms.MPE import mpe

        # evaluate acc of train and valid
        print("evaluating acc of train")
        train_class = train_data.copy()
        train_class[:, 0] = np.nan
        train_mpe = mpe(spn, train_data)
        class_acc = (train_mpe[:, 0] == train_data[:, 0]).mean()
        mlflow.log_metric("mspn_train_class_acc", class_acc)

        print("evaluating acc of valid")
        valid_targets = valid_ds.tensors[1].cpu().detach().numpy()
        valid_embeddings = model.get_embeddings(valid_dl, device)
        valid_data = valid_embeddings.cpu().detach().numpy()
        valid_data = np.concatenate([valid_targets[:, None], valid_data], axis=1)
        valid_class = valid_data.copy()
        valid_class[:, 0] = np.nan
        valid_mpe = mpe(spn, valid_data)
        class_acc = (valid_mpe[:, 0] == valid_data[:, 0]).mean()
        mlflow.log_metric("mspn_valid_class_acc", class_acc)

        # evaluate LLs
        print("evaluating ll of train")
        from spn.gpu.TensorFlow import eval_tf

        lltf = eval_tf(spn, train_data)
        mlflow.log_metric("mspn_train_ll", lltf.mean())

        print("evaluating ll of valid")
        lltf = eval_tf(spn, valid_data)
        mlflow.log_metric("mspn_valid_ll", lltf.mean())

        # evaluate
        model.eval()

        del train_dl
        del valid_dl

        # qualitative evaluation
        # mnist_expl_qualitative_evaluation(model, device)

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

            test_dl_rot_targets = test_ds_rot_s.tensors[1].cpu().detach().numpy()
            test_dl_rot_embeddings = model.get_embeddings(test_dl_rot, device)
            test_dl_rot_data = test_dl_rot_embeddings.cpu().detach().numpy()
            test_dl_rot_data = np.concatenate(
                [test_dl_rot_targets[:, None], test_dl_rot_data], axis=1
            )
            test_dl_rot_class = test_dl_rot_data.copy()
            test_dl_rot_class[:, 0] = np.nan
            test_dl_rot_mpe = mpe(spn, test_dl_rot_data)
            acc = (test_dl_rot_mpe[:, 0] == test_dl_rot_data[:, 0]).mean()

            ll_marg = eval_tf(spn, test_dl_rot_data).mean()
            test_dl_rot_marg = test_dl_rot_data.copy()
            test_dl_rot_marg[:, 0] = np.nan # label
            test_dl_rot_marg[:, 1] = np.nan # expl var 1 (rotation)
            ll_marg_rot =  eval_tf(spn, test_dl_rot_marg).mean()
            # TODO:
            expl_ll = 

            # acc = model.eval_acc(test_dl_rot, device)
            # ll_marg = model.eval_ll_marg(None, device, test_dl_rot)
            expl_ll = model.explain_ll(test_dl_rot, device)
            expl_mpe = model.explain_mpe(test_dl_rot, device)
            # convert to list
            expl_mpe = expl_mpe.tolist()
            if "rotation" not in eval_dict:
                eval_dict["rotation"] = {}
            eval_dict["rotation"][severity] = {
                "acc": acc,
                "ll_marg": ll_marg,
                "expl_ll": expl_ll,
                "expl_mpe": expl_mpe,
            }

            test_ds_cutoff_s = TensorDataset(
                cut_ds[prev_s:s], test_ds.targets[prev_s:s]
            )
            test_dl_cutoff = DataLoader(
                test_ds_cutoff_s, batch_size=batch_sizes["resnet"], shuffle=True
            )

            acc = model.eval_acc(test_dl_cutoff, device)
            ll_marg = model.eval_ll_marg(None, device, test_dl_cutoff)
            expl_ll = model.explain_ll(test_dl_cutoff, device)
            expl_mpe = model.explain_mpe(test_dl_cutoff, device)
            expl_mpe = expl_mpe.tolist()
            if "cutoff" not in eval_dict:
                eval_dict["cutoff"] = {}
            eval_dict["cutoff"][severity] = {
                "acc": acc,
                "ll_marg": ll_marg,
                "expl_ll": expl_ll,
                "expl_mpe": expl_mpe,
            }

            test_ds_noise_s = TensorDataset(
                noise_ds[prev_s:s], test_ds.targets[prev_s:s]
            )
            test_dl_noise = DataLoader(
                test_ds_noise_s, batch_size=batch_sizes["resnet"], shuffle=True
            )

            acc = model.eval_acc(test_dl_noise, device)
            ll_marg = model.eval_ll_marg(None, device, test_dl_noise)
            expl_ll = model.explain_ll(test_dl_noise, device)
            expl_mpe = model.explain_mpe(test_dl_noise, device)
            expl_mpe = expl_mpe.tolist()
            if "noise" not in eval_dict:
                eval_dict["noise"] = {}
            eval_dict["noise"][severity] = {
                "acc": acc,
                "ll_marg": ll_marg,
                "expl_ll": expl_ll,
                "expl_mpe": expl_mpe,
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
        overall_ll_marg = np.mean(
            [
                eval_dict[m][severity]["ll_marg"]
                for m in eval_dict
                for severity in eval_dict[m]
            ]
        )
        mlflow.log_metric("manip acc", overall_acc)
        mlflow.log_metric("manip ll marg", overall_ll_marg)

        # create plot for each corruption
        # x axis: severity
        # y axis: acc, ll, var, entropy

        # deactivate grid in plots
        plt.grid(False)
        plt.rcParams["axes.grid"] = False

        def create_expl_plot(corruption, mode, expl, accs, lls_marg):
            fig, ax = plt.subplots()

            ax.set_xlabel("severity")
            ax.set_xticks(np.array(list(range(5))) + 1)

            ax.plot(accs, label="accuracy", color="red")
            ax.set_ylabel("accuracy", color="red")
            ax.tick_params(axis="y", labelcolor="red")
            ax.set_ylim([0, 1])
            # ax2.set_ylim([0, 1])

            ax2 = ax.twinx()
            expl_tensor = torch.tensor(expl)
            ax2.plot(expl_tensor[:, 0], label=f"{mode} expl rot", color="green")
            ax2.plot(expl_tensor[:, 1], label=f"{mode} expl cut", color="purple")
            ax2.plot(expl_tensor[:, 2], label=f"{mode} expl noise", color="orange")
            ax2.tick_params(axis="y")
            ax2.set_ylabel(f"{mode} explanations")
            # if "mpe" in mode:
            #     ax2.set_ylim([0, 5])
            # else:
            #     ax2.set_ylim([0, 20])

            ax3 = ax.twinx()
            ax3.plot(lls_marg, label=f"lls marg", color="blue")
            ax3.set_ylabel("lls marg", color="blue")
            ax3.tick_params(axis="y", labelcolor="blue")

            fig.tight_layout()
            fig.legend(loc="upper left")
            mlflow.log_figure(fig, f"{corruption}_{mode}_expl.png")
            plt.close()

        for m in ["rotation", "cutoff", "noise"]:
            accs = [
                eval_dict[m][severity]["acc"]
                for severity in sorted(eval_dict[m].keys())
            ]
            ll_expl = [
                eval_dict[m][severity]["expl_ll"]
                for severity in sorted(eval_dict[m].keys())
            ]
            mpe_expl = [
                eval_dict[m][severity]["expl_mpe"]
                for severity in sorted(eval_dict[m].keys())
            ]
            lls_marg = [
                eval_dict[m][severity]["ll_marg"]
                for severity in sorted(eval_dict[m].keys())
            ]
            create_expl_plot(m, "ll", ll_expl, accs, lls_marg)
            create_expl_plot(m, "mpe", mpe_expl, accs, lls_marg)

        return lowest_val_loss
