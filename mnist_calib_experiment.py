import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import os
import mlflow

torch.manual_seed(0)

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

mlflow.set_experiment("mnist-calib")

data_dir = "/data_docker/datasets/"

# load data
mnist_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.reshape(-1, 28 * 28).squeeze()),
    ]
)

# load mnist
train_ds = datasets.MNIST(
    data_dir + "mnist",
    train=True,
    transform=mnist_transform,
    download=True,
)
valid_ds = datasets.MNIST(
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

print("manipulating images")
severity_levels = 5

# if we have 5k images, 1k per severity level
index_1 = len(test_ds) // severity_levels
index_2 = index_1 * 2
index_3 = index_1 * 3
index_4 = index_1 * 4
index_5 = index_1 * 5


def get_severity(i):
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


# 1: 20, 2: 40, 3: 60, 4: 80, 5: 100 degrees
rotations = torch.zeros((len(test_ds),))
for i, r in enumerate(rotations):
    severity = get_severity(i)
    rotations[i] = severity * 20

# 1: 2, 2: 4, 3: 9, 4: 16, 5: 25 pixels
cutoffs = torch.zeros((len(test_ds),))
for i, r in enumerate(cutoffs):
    severity = get_severity(i)
    cutoffs[i] = 2 if severity == 1 else severity**2

# 0: 10, 1: 20, 2: 30, 3: 40, 4: 50 noise
noises = torch.zeros((len(test_ds),))
for i, r in enumerate(noises):
    severity = get_severity(i)
    noises[i] = severity * 10

test_ds_rot = torch.zeros((len(test_ds), 28 * 28))
test_ds_cutoff = torch.zeros((len(test_ds), 28 * 28))
test_ds_noise = torch.zeros((len(test_ds), 28 * 28))
for i, img in enumerate(test_ds.data):
    image = img.reshape(28, 28, 1).clone()

    this_noise = torch.randn((28, 28, 1)) * noises[i]
    img_noise = torch.clamp(image + this_noise, 0, 255).to(dtype=torch.uint8)
    img_noise = transforms.ToTensor()(img_noise.numpy())
    test_ds_noise[i] = transforms.Normalize((0.1307,), (0.3081,))(img_noise).flatten()

    image_cutoff = image.clone()
    # cutoff rows
    image_cutoff[: int(cutoffs[i]), ...] = 0
    image_cutoff = transforms.ToTensor()(image_cutoff.numpy())
    test_ds_cutoff[i] = transforms.Normalize((0.1307,), (0.3081,))(
        image_cutoff
    ).flatten()

    image_rot = transforms.ToTensor()(image.numpy())
    image_rot = transforms.functional.rotate(
        img=image_rot, angle=int(rotations[i])  # , fill=-mean / std
    )
    test_ds_rot[i] = transforms.Normalize((0.1307,), (0.3081,))(image_rot).flatten()

print("done manipulating images")

train_ds, _ = torch.utils.data.random_split(
    train_ds, [0.8, 0.2], generator=torch.Generator().manual_seed(0)
)
_, valid_ds = torch.utils.data.random_split(
    valid_ds, [0.8, 0.2], generator=torch.Generator().manual_seed(0)
)


def start_run(run_name, batch_sizes, model_name, model_params, train_params):
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
        test_dl = DataLoader(test_ds, batch_size=batch_sizes["resnet"], shuffle=True)

        # create model
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

            resnet_spn = ConvResNetSPN(
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
            resnet_spn = ConvResnetDDU(
                block,
                layers,
                explaining_vars=[],  # for calibration test, we don't need explaining vars
                **model_params,
            )
        else:
            raise NotImplementedError
        mlflow.set_tag("model", resnet_spn.__class__.__name__)
        resnet_spn = resnet_spn.to(device)

        print("training resnet_spn")
        # train model
        resnet_spn.start_train(
            train_dl,
            valid_dl,
            device,
            checkpoint_dir=ckpt_dir,
            **train_params,
        )
        mlflow.pytorch.log_model(resnet_spn, "resnet_spn")
        # evaluate
        resnet_spn.eval()
        resnet_spn.einet_active = False
        train_acc = resnet_spn.eval_acc(train_dl, device)
        mlflow.log_metric("train accuracy resnet", train_acc)
        test_acc = resnet_spn.eval_acc(test_dl, device)
        mlflow.log_metric("test accuracy resnet", test_acc)

        resnet_spn.einet_active = True
        train_acc = resnet_spn.eval_acc(train_dl, device)
        mlflow.log_metric("train accuracy", train_acc)
        test_acc = resnet_spn.eval_acc(test_dl, device)
        mlflow.log_metric("test accuracy", test_acc)

        train_ll = resnet_spn.eval_ll(train_dl, device)
        mlflow.log_metric("train ll", train_ll)
        test_ll = resnet_spn.eval_ll(test_dl, device)
        mlflow.log_metric("test ll", test_ll)

        ood_ds_flat = ood_ds.data.reshape(-1, 28 * 28).to(dtype=torch.float32)
        ood_ds_flat = TensorDataset(ood_ds_flat, ood_ds.targets)
        ood_dl = DataLoader(ood_ds_flat, batch_size=batch_sizes["resnet"], shuffle=True)
        ood_acc = resnet_spn.eval_acc(ood_dl, device)
        mlflow.log_metric("ood accuracy", ood_acc)
        ood_ll = resnet_spn.eval_ll(ood_dl, device)
        mlflow.log_metric("ood ll", ood_ll)

        from tqdm import tqdm

        # calibration test
        eval_dict = {}
        severity_indices = [index_1, index_2, index_3, index_4, index_5]
        for s in tqdm(severity_indices):
            test_ds_rot_s = TensorDataset(test_ds_rot[:s], test_ds.targets[:s])
            test_dl_rot = DataLoader(
                test_ds_rot_s, batch_size=batch_sizes["resnet"], shuffle=True
            )
            severity = get_severity(s - 1)
            acc = resnet_spn.eval_acc(test_dl_rot, device)
            ll = resnet_spn.eval_ll(test_dl_rot, device)
            pred_var = resnet_spn.eval_pred_variance(test_dl_rot, device)
            pred_ent = resnet_spn.eval_pred_entropy(test_dl_rot, device)
            if "rotation" not in eval_dict:
                eval_dict["rotation"] = {}
            eval_dict["rotation"][severity] = {
                "acc": acc,
                "ll": ll.item(),
                "var": pred_var.item(),
                "entropy": pred_ent.item(),
            }

            test_ds_cutoff_s = TensorDataset(test_ds_cutoff[:s], test_ds.targets[:s])
            test_dl_cutoff = DataLoader(
                test_ds_cutoff_s, batch_size=batch_sizes["resnet"], shuffle=True
            )
            severity = get_severity(s - 1)
            acc = resnet_spn.eval_acc(test_dl_cutoff, device)
            ll = resnet_spn.eval_ll(test_dl_cutoff, device)
            pred_var = resnet_spn.eval_pred_variance(test_dl_cutoff, device)
            pred_ent = resnet_spn.eval_pred_entropy(test_dl_cutoff, device)
            if "cutoff" not in eval_dict:
                eval_dict["cutoff"] = {}
            eval_dict["cutoff"][severity] = {
                "acc": acc,
                "ll": ll.item(),
                "var": pred_var.item(),
                "entropy": pred_ent.item(),
            }

            test_ds_noise_s = TensorDataset(test_ds_noise[:s], test_ds.targets[:s])
            test_dl_noise = DataLoader(
                test_ds_noise_s, batch_size=batch_sizes["resnet"], shuffle=True
            )
            severity = get_severity(s - 1)
            acc = resnet_spn.eval_acc(test_dl_noise, device)
            ll = resnet_spn.eval_ll(test_dl_noise, device)
            pred_var = resnet_spn.eval_pred_variance(test_dl_noise, device)
            pred_ent = resnet_spn.eval_pred_entropy(test_dl_noise, device)
            if "noise" not in eval_dict:
                eval_dict["noise"] = {}
            eval_dict["noise"][severity] = {
                "acc": acc,
                "ll": ll.item(),
                "var": pred_var.item(),
                "entropy": pred_ent.item(),
            }

        mlflow.log_dict(eval_dict, "eval_dict")

        import numpy as np

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
        overall_var = np.mean(
            [
                eval_dict[m][severity]["var"]
                for m in eval_dict
                for severity in eval_dict[m]
            ]
        )
        overall_ent = np.mean(
            [
                eval_dict[m][severity]["entropy"]
                for m in eval_dict
                for severity in eval_dict[m]
            ]
        )
        mlflow.log_metric("manip acc", overall_acc)
        mlflow.log_metric("manip ll", overall_ll)
        mlflow.log_metric("manip var", overall_var)
        mlflow.log_metric("manip ent", overall_ent)

        # create plot for each corruption
        # x axis: severity
        # y axis: acc, ll, var, entropy
        import matplotlib.pyplot as plt

        for m in ["rotation", "cutoff", "noise"]:
            accs = [
                eval_dict[m][severity]["acc"]
                for severity in sorted(eval_dict[m].keys())
            ]
            lls = [
                eval_dict[m][severity]["ll"] for severity in sorted(eval_dict[m].keys())
            ]
            vars = [
                eval_dict[m][severity]["var"]
                for severity in sorted(eval_dict[m].keys())
            ]
            ents = [
                eval_dict[m][severity]["entropy"]
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
            ax3.plot(vars, label="pred var", color="green")
            # ax3.set_ylabel("predictive variance", color="green")
            ax3.tick_params(axis="y", labelcolor="green")

            ax4 = ax.twinx()
            ax4.plot(ents, label="pred entropy", color="orange")
            # ax4.set_ylabel("predictive entropy", color="orange")
            ax4.tick_params(axis="y", labelcolor="orange")

            fig.tight_layout()
            fig.legend()
            mlflow.log_figure(fig, f"{m}.png")


model_params = dict(
    block="bottleneck",
    layers=[2, 2, 2, 2],
    num_classes=10,
    image_shape=(1, 28, 28),
    einet_depth=3,
    einet_num_sums=20,
    einet_num_leaves=20,
    einet_num_repetitions=1,
    einet_leaf_type="Normal",
    einet_dropout=0.0,
    spec_norm_bound=0.9,  # only for ConvResNetSPN
    spectral_normalization=True,  # only for ConvResNetDDU
    mod=True,  # only for ConvResNetDDU
)
train_params = dict(
    learning_rate_warmup=0.05,
    learning_rate=0.05,
    lambda_v=0.995,
    warmup_epochs=3,
    num_epochs=3,
    deactivate_resnet=True,
    lr_schedule_warmup_step_size=10,
    lr_schedule_warmup_gamma=0.5,
    lr_schedule_step_size=10,
    lr_schedule_gamma=0.5,
    early_stop=10,
)
run_name = "seperate"
batch_sizes = dict(resnet=512)
model_name = "ConvResNetSPN"
# model_name = "ConvResNetDDU"
start_run(run_name, batch_sizes, model_name, model_params, train_params)
