# %%
eval_dicts = {
    "cifar10-c-calib": "279248034225110540/506d9d588d0f459680ca8a4108f85036/artifacts/eval_dict",
    "cifar10-c-expl": "298139550611321154/8338ffec57654ad4ada228e5b8b48979/artifacts/eval_dict",
    "svhn-c-calib": "382722780317026903/78d91c5e24624425af327005aade7358/artifacts/eval_dict",
    "svhn-c-expl": "354955436886369284/b90c9c1d94cb41dd801ebea1528e6460/artifacts/eval_dict",
}
pre_path = "/data_docker/mlartifacts/"

# %%
noise = ["gaussian_noise", "shot_noise", "impulse_noise"]
blur = ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"]
weather = ["snow", "frost", "fog"]
digital = [
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]
corruptions = noise + blur + weather + digital

# %% [markdown]
# # Corruption plots

# %%
import json

svhn_calib_path = pre_path + eval_dicts["svhn-c-calib"]
svhn_calib = {}
with open(svhn_calib_path) as json_file:
    svhn_calib = json.load(json_file)

cifar10_calib_path = pre_path + eval_dicts["cifar10-c-calib"]
cifar10_calib = {}
with open(cifar10_calib_path) as json_file:
    cifar10_calib = json.load(json_file)

dataset = "CIFAR10"
if dataset == "SVHN":
    eval_dict = svhn_calib
elif dataset == "CIFAR10":
    eval_dict = cifar10_calib

# %%
import numpy as np
import torch

# %%
noise_accs = []
noise_lls = []
noise_ent = []
blur_accs = []
blur_lls = []
blur_ent = []
weather_accs = []
weather_lls = []
weather_ent = []
digital_accs = []
digital_lls = []
digital_ent = []

for corruption in corruptions:
    backbone_accs = [
        eval_dict[corruption][severity]["backbone_acc"]
        for severity in eval_dict[corruption]
    ]
    einet_accs = [
        eval_dict[corruption][severity]["einet_acc"]
        for severity in eval_dict[corruption]
    ]
    lls_marg = [
        eval_dict[corruption][severity]["ll_marg"] for severity in eval_dict[corruption]
    ]
    entropy = [
        eval_dict[corruption][severity]["entropy"] for severity in eval_dict[corruption]
    ]
    if corruption in noise:
        noise_accs.append(backbone_accs)
        noise_lls.append(lls_marg)
        noise_ent.append(entropy)
    elif corruption in blur:
        blur_accs.append(backbone_accs)
        blur_lls.append(lls_marg)
        blur_ent.append(entropy)
    elif corruption in weather:
        weather_accs.append(backbone_accs)
        weather_lls.append(lls_marg)
        weather_ent.append(entropy)
    elif corruption in digital:
        digital_accs.append(backbone_accs)
        digital_lls.append(lls_marg)
        digital_ent.append(entropy)
noise_accs = np.array(noise_accs)
noise_lls = np.array(noise_lls)
noise_ent = np.array(noise_ent)
blur_accs = np.array(blur_accs)
blur_lls = np.array(blur_lls)
blur_ent = np.array(blur_ent)
weather_accs = np.array(weather_accs)
weather_lls = np.array(weather_lls)
weather_ent = np.array(weather_ent)
digital_accs = np.array(digital_accs)
digital_lls = np.array(digital_lls)
digital_ent = np.array(digital_ent)

# %%
from plotting_utils import uncert_corrupt_plot
import matplotlib.pyplot as plt

# fig = uncert_corrupt_plot(
#     noise_accs.mean(axis=0), noise_lls.mean(axis=0), "Noise Corruptions", mode="ll"
# )
# fig.show()
# fig.savefig(f"noise_ll_{dataset}.pdf")

# fig = uncert_corrupt_plot(
#     blur_accs.mean(axis=0), blur_lls.mean(axis=0), "Blur Corruptions", mode="ll"
# )
# fig.show()
# fig.savefig(f"blur_ll_{dataset}.pdf")

# fig = uncert_corrupt_plot(
#     weather_accs.mean(axis=0), weather_lls.mean(axis=0), "Weather Corruptions", mode="ll"
# )
# fig.show()
# fig.savefig(f"weather_ll_{dataset}.pdf")

# fig = uncert_corrupt_plot(
#     digital_accs.mean(axis=0), digital_lls.mean(axis=0), "Digital Corruptions", mode="ll"
# )
# fig.show()
# fig.savefig(f"digital_ll_{dataset}.pdf")

fig = uncert_corrupt_plot(
    noise_accs.mean(axis=0), noise_ent.mean(axis=0), "Noise Corruptions", mode="ent"
)
fig.show()
fig.savefig(f"noise_ent_{dataset}.pdf")

fig = uncert_corrupt_plot(
    blur_accs.mean(axis=0), blur_ent.mean(axis=0), "Blur Corruptions", mode="ent"
)
fig.show()
fig.savefig(f"blur_ent_{dataset}.pdf")

fig = uncert_corrupt_plot(
    weather_accs.mean(axis=0),
    weather_ent.mean(axis=0),
    "Weather Corruptions",
    mode="ent",
)
fig.show()
fig.savefig(f"weather_ent_{dataset}.pdf")

fig = uncert_corrupt_plot(
    digital_accs.mean(axis=0),
    digital_ent.mean(axis=0),
    "Digital Corruptions",
    mode="ent",
)
fig.show()
fig.savefig(f"digital_ent_{dataset}.pdf")


# %% [markdown]
# # Calibration plots

# %%
import torch
import numpy as np


def load_cifar100c(test_transformer, test_ds):
    dataset_dir = "/data_docker/datasets/"
    # cifar100_c_url = "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1"
    cifar100_c_path = "CIFAR-100-C"
    cifar100_c_path_complete = dataset_dir + cifar100_c_path
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
    from tqdm import tqdm

    print("loading all corrupted data")
    cifar100_c_ds = torch.zeros((10000 * len(corruptions) * 5, 32 * 32 * 3))
    index = 0
    for corruption in tqdm(corruptions):
        curr_cifar100 = np.load(f"{cifar100_c_path_complete}/{corruption}.npy")
        curr_cifar100 = torch.stack(
            [test_transformer(img) for img in curr_cifar100], dim=0
        )
        cifar100_c_ds[index : index + 10000 * 5] = curr_cifar100
        index += 10000 * 5
    targets = torch.cat(
        [torch.tensor(test_ds.targets) for _ in range(len(corruptions) * 5)], dim=0
    )

    print("shapes of corrupted stuff: ", cifar100_c_ds.shape, targets.shape)
    cifar100_c_ds = list(
        zip(
            cifar100_c_ds.to(dtype=torch.float32),
            targets.reshape(-1),
        )
    )

    cifar100_c_dl = torch.utils.data.DataLoader(
        cifar100_c_ds,
        batch_size=512,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )
    return cifar100_c_dl


def load_cifar100():
    from cifar100_calib_experiment import load_datasets

    _, _, test_ds, transformer = load_datasets()
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=512, shuffle=False, num_workers=4
    )
    test_c_dl = load_cifar100c(transformer, test_ds)

    return test_dl, test_c_dl


def load_cifar10c(test_transformer, test_ds):
    dataset_dir = "/data_docker/datasets/"
    # cifar10_c_url = "https://zenodo.org/records/3555552/files/CIFAR-10-C.tar?download=1"
    cifar10_c_path = "CIFAR-10-C"
    cifar10_c_path_complete = dataset_dir + cifar10_c_path
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

    cifar10_c_dl = torch.utils.data.DataLoader(
        cifar10_c_ds,
        batch_size=512,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )
    return cifar10_c_dl


def load_cifar10():
    from cifar10_calib_experiment import load_datasets

    _, _, test_ds, transformer = load_datasets()
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=512, shuffle=False, num_workers=4
    )
    test_c_dl = load_cifar10c(transformer, test_ds)
    return test_dl, test_c_dl


def load_svhn_c(transformer, test_ds):
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
    from tqdm import tqdm

    print("loading all corrupted data")
    num_samples = 26032
    svhn_c_ds = torch.zeros((num_samples * len(corruptions) * 5, 32 * 32 * 3))
    levels = [1, 2, 3, 4, 5]
    index = 0
    for corruption in tqdm(corruptions):
        for l in levels:
            curr_svhn = np.load(
                f"{svhn_c_path_complete}/svhn_test_{corruption}_l{l}.npy"
            )
            curr_svhn = torch.stack([transformer(img) for img in curr_svhn], dim=0)
            svhn_c_ds[index : index + num_samples] = curr_svhn
            index += num_samples
    targets = torch.cat(
        [torch.tensor(test_ds.labels) for _ in range(len(corruptions) * 5)], dim=0
    )

    print("shapes of corrupted stuff: ", svhn_c_ds.shape, targets.shape)
    svhn_c_ds = list(
        zip(
            svhn_c_ds.to(dtype=torch.float32),
            targets.reshape(-1),
        )
    )

    svhn_c_dl = torch.utils.data.DataLoader(
        svhn_c_ds,
        batch_size=512,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )
    return svhn_c_dl


def load_svhn():
    from svhn_calib_experiment import load_datasets

    _, _, test_ds, transformer = load_datasets()
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=512, shuffle=False, num_workers=4
    )
    test_c_dl = load_svhn_c(transformer, test_ds)
    return test_dl, test_c_dl


# %%
model_params = dict(
    explaining_vars=[],
    block="basic",  # basic, bottleneck
    layers=[2, 2, 2, 2],
    num_classes=10,
    image_shape=(3, 32, 32),
    einet_depth=5,
    einet_num_sums=11,
    einet_num_leaves=15,
    einet_num_repetitions=10,
    einet_leaf_type="Normal",
    einet_dropout=0.0,
    # spec_norm_bound=0.95,  # only for ConvResNetSPN
    spec_norm_bound=7,  # only for ConvResNetSPN
    spectral_normalization=True,  # only for ConvResNetDDU
    mod=True,  # only for ConvResNetDDU
    num_hidden=32,
)

# %%
# load dataset and their manipulations
svhn, svhn_c = load_svhn()
# cifar100, cifar100_c = load_cifar100()
# cifar10, cifar10_c = load_cifar10()

# %%
# dls = [cifar100, cifar100_c]
# dls = [cifar10, cifar10_c]
dls = {
    "svhn": svhn,
    # "svhn-c": svhn_c,
    # "cifar100": cifar100,
    # "cifar100-c": cifar100_c,
    # "cifar10": cifar10,
    # "cifar10-c": cifar10_c,
}

device = "cpu"
from plotting_utils import calibration_plot_multi

# load models from checkpoints
model_checkpoints_c10 = {
    "MCD": "279248034225110540/acf0f10905e24b8380b358a593174256/artifacts/model/state_dict.pth",
    "Softmax": "279248034225110540/3d630a5cb75044ff9cd098169bf24bc9/artifacts/model/state_dict.pth",
    # "SNGP": "279248034225110540/f3b109953e984d17ab6c342ade33ad1f/artifacts/model/state_dict.pth",
    "DDeXU": "279248034225110540/cafe642add1a45f3b2a8337d22d6ef52/artifacts/model/state_dict.pth",
    # "DDU": "279248034225110540/3d630a5cb75044ff9cd098169bf24bc9/artifacts/model/state_dict.pth",
    # "Ensemble": "354955436886369284/b90c9c1d94cb41dd801ebea1528e6460/artifacts/model",
    "DE": [
        "279248034225110540/df45c655ce064173bfa2bce97ea1873e/artifacts/model/state_dict.pth",
        "279248034225110540/e298816027c94fb0967ff67732b7ee85/artifacts/model/state_dict.pth",
        "279248034225110540/3d630a5cb75044ff9cd098169bf24bc9/artifacts/model/state_dict.pth",
        "279248034225110540/813b4f9b8b77449fbe35741917c8b371/artifacts/model/state_dict.pth",
        "279248034225110540/b2864c0fb59341e0acaf654448d0630c/artifacts/model/state_dict.pth",
    ],
}
model_checkpoints_svhn = {
    "MCD": "382722780317026903/06fc32a388344a21a204adb2920bb0cf/artifacts/model/state_dict.pth",
    "Softmax": "382722780317026903/a53d43877c7f41518e3c79e9b6cf4b0f/artifacts/model/state_dict.pth",
    # "SNGP": "279248034225110540/f3b109953e984d17ab6c342ade33ad1f/artifacts/model/state_dict.pth",
    "DDeXU": "382722780317026903/3ca2eea2cc534d329f7aa2134bbc5aa8/artifacts/model/state_dict.pth",
    # "DDU": "279248034225110540/3d630a5cb75044ff9cd098169bf24bc9/artifacts/model/state_dict.pth",
    # "Ensemble": "354955436886369284/b90c9c1d94cb41dd801ebea1528e6460/artifacts/model",
    "DE": [
        "382722780317026903/580ba0e7f39f4eab96f6d68d35b73fff/artifacts/model/state_dict.pth",
        "382722780317026903/a53d43877c7f41518e3c79e9b6cf4b0f/artifacts/model/state_dict.pth",
        "382722780317026903/3f6af23c46d24eb5af67eaa8444fe4c9/artifacts/model/state_dict.pth",
        "382722780317026903/ebde1178f0c8432084fc3ec07d55f839/artifacts/model/state_dict.pth",
        "382722780317026903/a8fae2c6ba994f8abb6024f108a2f2d6/artifacts/model/state_dict.pth",
    ],
}
model_checkpoints_c100 = {
    "MCD": "352663593920914837/fdc65a0fc9884ab98f8f7414901c11e0/artifacts/model/state_dict.pth",
    "Softmax": "352663593920914837/96677430286e4e9e9bfe8338217d23b4/artifacts/model/state_dict.pth",
    # "SNGP": "279248034225110540/f3b109953e984d17ab6c342ade33ad1f/artifacts/model/state_dict.pth",
    "DDeXU": "352663593920914837/12e2ae61d5c14affa319f9f9b4d19ac6/artifacts/model/state_dict.pth",
    # "DDU": "279248034225110540/3d630a5cb75044ff9cd098169bf24bc9/artifacts/model/state_dict.pth",
    "DE": [
        "352663593920914837/96677430286e4e9e9bfe8338217d23b4/artifacts/model/state_dict.pth",
        "352663593920914837/cd7d1bb9e08c4d7790c8dc56f099b1af/artifacts/model/state_dict.pth",
        "352663593920914837/6914054a73694a48902b7f50428bd373/artifacts/model/state_dict.pth",
        "352663593920914837/917f120246154a889a4989c0f0180948/artifacts/model/state_dict.pth",
        "352663593920914837/3adc5d76ce0f4288abc5f308d0860a15/artifacts/model/state_dict.pth",
    ],
}

all_model_checkpoints = {
    "cifar10": model_checkpoints_c10,
    "cifar10-c": model_checkpoints_c10,
    "cifar100": model_checkpoints_c100,
    "cifar100-c": model_checkpoints_c100,
    "svhn": model_checkpoints_svhn,
    "svhn-c": model_checkpoints_svhn,
}

from ResNetSPN import EfficientNetDet
from ResNetSPN import EfficientNetSPN
from ResNetSPN import EfficientNetGMM
from ResNetSPN import EfficientNetEnsemble
from ResNetSPN import EfficientNetDropout
from ResNetSPN import EfficientNetSNGP

import mlflow

mlflow.set_tracking_uri("http://localhost:8080")
mlflow.set_experiment("Evaluation")
if mlflow.active_run() is not None:
    mlflow.end_run()
# run model.eval_calibration with the dataset for each model
for dl_name in dls:
    confs_freq = []
    accs_freq = []
    confs_w = []
    accs_w = []
    names = []
    model_checkpoints = all_model_checkpoints[dl_name]

    with mlflow.start_run(run_name=f"Eval_{dl_name}"):
        for model_name in model_checkpoints:
            print(model_name)
            if model_name == "Softmax":
                model_params["spectral_normalization"] = False
                model = EfficientNetDet(**model_params)
            elif model_name == "DDeXU":
                model = EfficientNetSPN(**model_params)
            elif model_name == "DDU":
                model = EfficientNetGMM(**model_params)
            elif model_name == "SNGP":
                model_params["spectral_normalization"] = True
                train_batch_size = (512,)
                train_num_data = 45000 + 5000
                model = EfficientNetSNGP(
                    train_batch_size=train_batch_size,
                    train_num_data=train_num_data,
                    **model_params,
                )
            elif model_name == "MCD":
                model = EfficientNetDropout(**model_params)
            elif model_name == "DE":
                model_params["ensemble_paths"] = [
                    "/data_docker/mlartifacts/" + m
                    for m in model_checkpoints[model_name]
                ]
                model = EfficientNetEnsemble(**model_params, map_location=device)
                for m in model.members:
                    m.to(device)
            if model_name != "DE":
                path = pre_path + model_checkpoints[model_name]
                model.load(path, backbone_only=False, map_location=device)
            model = model.to(device)
            model = model.eval()
            dl = dls[dl_name]
            if "DDeXU" in model_name:
                model.compute_normalization_values(
                    dl, device
                )  # this should be train instead of test
            model.activate_uncert_head(deactivate_backbone=True)
            (conf_freq, acc_freq), (conf_w, acc_w) = model.eval_calibration(
                None, device, model_name, dl
            )
            confs_freq.append(conf_freq)
            accs_freq.append(acc_freq)
            confs_w.append(conf_w)
            accs_w.append(acc_w)
            names.append(model_name)

        calibration_plot_multi(confs_freq, accs_freq, names, "freq", dl_name)
        calibration_plot_multi(confs_w, accs_w, names, "w", dl_name)

# %% [markdown]
# # Explanation plots

# %%

cifar10_expl_path = pre_path + eval_dicts["cifar10-c-expl"]
cifar10_expl = {}
with open(cifar10_expl_path) as json_file:
    cifar10_expl = json.load(json_file)
eval_dict = cifar10_expl
print(eval_dict)

corruptions_eval = ["snow", "defocus_blur", "speckle_noise", "pixelate"]

# %%
from plotting_utils import explain_plot

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
for corruption in eval_dict:
    if corruption not in corruptions_eval:
        continue
    accs = [eval_dict[corruption][l]["acc"] for l in eval_dict[corruption]]
    entropy = [eval_dict[corruption][l]["entropy"] for l in eval_dict[corruption]]
    lls_marg = [eval_dict[corruption][l]["ll_marg"] for l in eval_dict[corruption]]
    # get corruption index
    corr_idx = corruptions.index(corruption)
    expl_ll = [
        eval_dict[corruption][l]["expl_ll"] for l in eval_dict[corruption]
    ]  # list of list, shape: [num_levels, num_expl_vars]
    expl_ll = torch.tensor(expl_ll)
    # expl_ll.shape = [num_levels, num_expl_vars]
    expl_mpe = [eval_dict[corruption][l]["expl_mpe"] for l in eval_dict[corruption]]
    expl_mpe = torch.tensor(expl_mpe)

    expl_post = [eval_dict[corruption][l]["expl_post"] for l in eval_dict[corruption]]
    expl_post = torch.tensor(expl_post)
    print(len(corruptions))
    print(expl_ll.shape)
    show_legend = True if corruption == "pixelate" else False
    fig = explain_plot(
        corruptions, lls_marg, expl_ll, corruption, "ll", show_legend=show_legend
    )
    fig.savefig(f"expl_ll_{corruption}.pdf")
    fig = explain_plot(
        corruptions, entropy, expl_mpe, corruption, "mpe", show_legend=show_legend
    )
    fig.savefig(f"expl_mpe_{corruption}.pdf")
    fig = explain_plot(
        corruptions, entropy, expl_post, corruption, "post", show_legend=show_legend
    )
    fig.savefig(f"expl_post_{corruption}.pdf")

# %%
