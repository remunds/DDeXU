import traceback
import torch
import mlflow
import optuna

from simple_einet.layers.distributions.multidistribution import MultiDistributionLayer
from simple_einet.layers.distributions.normal import RatNormal
from two_moons_experiment import start_two_moons_run
from mnist_calib_experiment import start_mnist_calib_run
from figure6 import start_figure6_run
from figure6_latent import start_latent_figure6_run

# from mnist_expl_experiment import start_mnist_expl_run, mnist_expl_manual_evaluation

from mnist_expl_experiment2 import start_mnist_expl_run, mnist_expl_manual_evaluation
from dirty_mnist_experiment import start_dirty_mnist_run

from cifar10_expl_experiment import start_cifar10_expl_run
from cifar10_expl_brightness import start_cifar10_brightness_run
from cifar10_calib_experiment import start_cifar10_calib_run
from svhn_calib_experiment import start_svhn_calib_run
from svhn_expl_experiment import start_svhn_expl_run

# from svhn_expl_experiment2 import start_svhn_expl_run
# from svhn_expl_experiment_einsum import start_svhn_expl_run
# from svhn_expl_experiment_custom import start_svhn_expl_run
# from svhn_expl_experiment_pres import start_svhn_expl_run

torch.manual_seed(0)
# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")


def suggest_hps(trial, train_params, model_params):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)

    train_params["learning_rate"] = lr

    return train_params, model_params


def suggest_hps_backbone_only(trial, train_params, model_params):
    train_params["learning_rate"] = 1  # irrelevant
    train_params["learning_rate_warmup"] = trial.suggest_float(
        "lr_warmup", 1e-4, 1e-1, log=True
    )

    return train_params, model_params


def suggest_hps_einet_only(trial, train_params, model_params):
    train_params["learning_rate_warmup"] = 1  # irrelevant
    train_params["learning_rate"] = trial.suggest_float(
        "lr_warmup", 1e-4, 1e-1, log=True
    )
    model_params["einet_depth"] = trial.suggest_categorical("einet_depth", [3, 5])
    model_params["einet_num_repetitions"] = trial.suggest_categorical(
        "einet_rep", [1, 5]
    )
    model_params["einet_dropout"] = trial.suggest_categorical(
        "einet_dropout", [0.0, 0.2]
    )

    return train_params, model_params


def tune_two_moons(loss, training, pretrained_path=None):
    print("New tuning run of two moons")
    run_name = f"{loss}_{training}"

    def objective(trial):
        batch_sizes = dict(resnet=512)
        train_params = dict(
            warmup_epochs=100,
            num_epochs=100,
            early_stop=5,
        )
        model_params_dense = dict(
            input_dim=2,
            output_dim=2,
            num_layers=3,
            num_hidden=32,
            spec_norm_bound=0.95,
            einet_depth=3,  # might be overwritten by optuna
            einet_num_sums=20,
            einet_num_leaves=20,
            einet_num_repetitions=1,  # might be overwritten by optuna
            einet_leaf_type="Normal",
            einet_dropout=0.0,
        )
        if loss == "discriminative" or loss == "noloss":
            train_params["lambda_v"] = 1.0
        elif loss == "generative":
            train_params["lambda_v"] = 0.0
        elif loss == "hybrid":
            train_params["lambda_v"] = 0.5
        elif loss == "hybrid_low":
            train_params["lambda_v"] = 0.1
        elif loss == "hybrid_very_low":
            train_params["lambda_v"] = 0.01
        elif loss == "hybrid_high":
            train_params["lambda_v"] = 0.9
        elif loss == "hybrid_very_high":
            train_params["lambda_v"] = 0.99
        else:
            raise ValueError(
                "loss must be discriminative, generative, hybrid, hybrid_low, hybrid_very_low or hybrid_high"
            )

        if training == "end-to-end":
            train_params["warmup_epochs"] = 0
            train_params["deactivate_backbone"] = False
        elif training == "seperate":
            train_params["warmup_epochs"] = 100
            train_params["deactivate_backbone"] = True
        elif training == "warmup":
            train_params["warmup_epochs"] = 100
            train_params["deactivate_backbone"] = False
        elif training == "backbone_only":
            train_params["warmup_epochs"] = 50
            train_params["deactivate_backbone"] = False
            train_params["num_epochs"] = 0
        elif training == "einet_only":
            train_params["warmup_epochs"] = 0
            train_params["deactivate_backbone"] = True
            train_params["num_epochs"] = 100
        else:
            raise ValueError(
                "training must be end-to-end, seperate, warmup, backbone_only or einet_only"
            )
        if pretrained_path is not None:
            train_params["pretrained_path"] = pretrained_path
        if training == "backbone_only":
            train_params, model_params_dense = suggest_hps_backbone_only(
                trial, train_params, model_params_dense
            )
        elif training == "einet_only":
            train_params, model_params_dense = suggest_hps_einet_only(
                trial, train_params, model_params_dense
            )
        else:
            train_params, model_params_dense = suggest_hps(
                trial, train_params, model_params_dense
            )
        try:
            return start_two_moons_run(
                run_name, batch_sizes, model_params_dense, train_params, trial
            )
        except Exception as e:
            if type(e) == optuna.exceptions.TrialPruned:
                raise e
            print(e)
            traceback.print_exc()
            mlflow.set_tag("pruned", e)
            mlflow.end_run()
            raise optuna.TrialPruned()

    dataset = "two-moons"
    exp = mlflow.get_experiment_by_name(dataset)
    runs = []
    if exp:
        query = f"attributes.run_name = '{run_name}'"
        runs = mlflow.search_runs([exp.experiment_id], query)
    n_trials = 15
    # only run experiment, if it wasnt run fully
    if len(runs) < n_trials:
        mlflow.set_experiment(dataset)
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=3,  # requires at least 3 results to start pruning
                n_warmup_steps=10,  # trial needs to log at least 10 steps before pruning
            ),
        )
        study.optimize(objective, n_trials=n_trials)


def run_conv(dataset, loss, training, model, pretrained_path=None):
    print(f"New run of {dataset} with {loss} and {training} and {model}")
    mlflow.set_experiment(dataset)
    run_name = f"{loss}_{training}_{model}"

    batch_sizes = dict(resnet=512)
    if "mnist" in dataset:
        image_shape = (1, 28, 28)
    elif "cifar10" in dataset or "svhn" in dataset:
        image_shape = (3, 32, 32)
    else:
        raise ValueError(
            "dataset must be dirty-mnist, mnist-calib, mnist-expl or cifar10-c"
        )

    if "SPN" in model or "DDU" in model or "GMM" in model:
        model_params = dict(
            model=model,  # ConvResNetSPN, ConvResNetDDU
            block="basic",  # basic, bottleneck
            layers=[2, 2, 2, 2],
            num_classes=10,
            image_shape=image_shape,
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
        )
    elif "SNGP" in model or "Dropout" in model or "Ensemble" in model:
        model_params = dict(
            model=model,
            num_classes=10,
            image_shape=image_shape,
            train_batch_size=batch_sizes["resnet"],
            spec_norm_bound=7,
            num_hidden=64,  # 16 too low, 32 worked well, failed with 128 already
            # spec_norm_bound=20,  # i think 7 might be good for cifar etc.
        )
    train_params = dict(
        pretrained_path=pretrained_path,
        # learning_rate_warmup=0.035,
        learning_rate_warmup=0.03,
        early_stop=30,
    )
    if loss == "discriminative" or loss == "noloss":
        train_params["lambda_v"] = 1.0
    elif loss == "generative":
        train_params["lambda_v"] = 0.0
    elif loss == "hybrid":
        train_params["lambda_v"] = 0.5
    elif loss == "hybrid_low":
        train_params["lambda_v"] = 0.1
    elif loss == "hybrid_mid_low":
        train_params["lambda_v"] = 0.3
    elif loss == "hybrid_high":
        train_params["lambda_v"] = 0.9
    elif loss == "hybrid_mid_high":
        train_params["lambda_v"] = 0.7
    else:
        raise ValueError(
            "loss must be discriminative, generative, hybrid, hybrid_low, hybrid_very_low or hybrid_high"
        )

    if training == "end-to-end":
        train_params["warmup_epochs"] = 0
        train_params["deactivate_backbone"] = False
        train_params["num_epochs"] = 800
    elif training == "seperate":
        train_params["warmup_epochs"] = 400
        train_params["deactivate_backbone"] = True
        train_params["num_epochs"] = 400
    elif training == "warmup":
        train_params["warmup_epochs"] = 50
        train_params["deactivate_backbone"] = False
    elif training == "backbone_only":
        train_params["warmup_epochs"] = 400
        train_params["deactivate_backbone"] = False
        train_params["num_epochs"] = 0
    elif training == "einet_only":
        train_params["warmup_epochs"] = 0
        train_params["deactivate_backbone"] = True
        train_params["num_epochs"] = 400
    elif training == "eval_only":
        train_params["warmup_epochs"] = 0
        train_params["deactivate_backbone"] = True
        train_params["num_epochs"] = 0
    else:
        raise ValueError(
            "training must be end-to-end, seperate, warmup, backbone_only or einet_only"
        )
    if pretrained_path is not None:
        train_params["pretrained_path"] = pretrained_path
    if model == "ConvResNetSPN":
        lr = 0.002
    elif "ConvResNetDDU" in model:
        lr = 0.02
    # elif "EfficientNetSNGP" in model:
    #     lr = 0.005
    elif "EfficientNet" in model:
        lr = 0.015
        # lr = 0.02
        # lr = 0.05
    else:
        raise ValueError(
            "model must be ConvResNetSPN, ConvResNetDDU or EfficientNetSPN"
        )
    train_params["learning_rate"] = lr
    if "mnist-calib" in dataset:
        try:
            start_mnist_calib_run(
                run_name, batch_sizes, model_params, train_params, None
            )
        except Exception as e:
            print(e)
            traceback.print_exc()
            mlflow.set_tag("pruned", e)
            mlflow.end_run()
    elif "mnist-expl" in dataset:
        model_params["explaining_vars"] = [0, 1, 2]  # rotations, cutoffs, noises
        # train_params["highest_severity_train"] = 2
        train_params["highest_severity_train"] = 5
        # copy here, since some params are changed in the experiment
        # train_params["use_mpe_reconstruction_loss"] = True
        try:
            start_mnist_expl_run(
                run_name,
                batch_sizes.copy(),
                model_params.copy(),
                train_params.copy(),
                None,
            )
        except Exception as e:
            print(e)
            traceback.print_exc()
            mlflow.set_tag("pruned", e)
            mlflow.end_run()
    elif "dirty-mnist" in dataset:
        try:
            start_dirty_mnist_run(
                run_name, batch_sizes, model_params, train_params, None
            )
        except Exception as e:
            print(e)
            traceback.print_exc()
            mlflow.set_tag("pruned", e)
            mlflow.end_run()
    elif "cifar10-c-calib" in dataset:
        try:
            start_cifar10_calib_run(
                run_name, batch_sizes, model_params, train_params, None
            )
        except Exception as e:
            print(e)
            traceback.print_exc()
            mlflow.set_tag("pruned", e)
            mlflow.end_run()
    elif "cifar10-c-expl" in dataset:
        model_params["explaining_vars"] = list(range(19))
        train_params["corruption_levels_train"] = [0, 1]
        # train_params["use_mpe_reconstruction_loss"] = True
        try:
            start_cifar10_expl_run(
                run_name,
                batch_sizes.copy(),
                model_params.copy(),
                train_params.copy(),
                None,
            )
        except Exception as e:
            print(e)
            traceback.print_exc()
            mlflow.set_tag("pruned", e)
            mlflow.end_run()
    elif "cifar10-expl-bright" in dataset:

        def start_bright_run():
            try:
                start_cifar10_brightness_run(
                    run_name,
                    batch_sizes.copy(),
                    model_params.copy(),
                    train_params.copy(),
                    None,
                )
            except Exception as e:
                print(e)
                traceback.print_exc()
                mlflow.set_tag("pruned", e)
                mlflow.end_run()

        model_params["explaining_vars"] = list(range(1))
        train_params["corruption_levels_train"] = [0, 1]
        # for a in [True, False]:
        #     for b in [128, 256]:
        # train_params["train_backbone_default"] = True
        # train_params["use_mpe_reconstruction_loss"] = a
        # model_params["num_hidden"] = b
        # start_bright_run()
        train_params["train_backbone_default"] = True
        # train_params["use_mpe_reconstruction_loss"] = True
        train_params["use_mpe_reconstruction_loss"] = True
        model_params["num_hidden"] = 128
        start_bright_run()

    elif "svhn-c-calib" in dataset:
        try:
            start_svhn_calib_run(
                run_name, batch_sizes, model_params, train_params, None
            )
        except Exception as e:
            print(e)
            traceback.print_exc()
            mlflow.set_tag("pruned", e)
            mlflow.end_run()
    elif "svhn-c-expl" in dataset:
        model_params["explaining_vars"] = list(range(15))
        train_params["corruption_levels_train"] = [0, 1]
        # train_params["use_mpe_reconstruction_loss"] = True
        try:
            start_svhn_expl_run(
                run_name,
                batch_sizes.copy(),
                model_params.copy(),
                train_params.copy(),
                None,
            )
        except Exception as e:
            print(e)
            traceback.print_exc()
            mlflow.set_tag("pruned", e)
            mlflow.end_run()

    else:
        raise ValueError(
            "dataset must be mnist-calib, mnist-expl, dirty-mnist, cifar10-c or cifar10-c_expl"
        )


def tune_conv(dataset, loss, training, model, pretrained_path=None):
    print(f"New tuning run of {dataset} with {loss} and {training} and {model}")
    run_name = f"{loss}_{training}_{model}"

    def objective(trial):
        batch_sizes = dict(resnet=512)
        train_params = dict(
            learning_rate_warmup=0.05,  # irrelevant for end-to-end
            warmup_epochs=100,
            num_epochs=100,
            early_stop=10,
        )
        if "mnist" in dataset:
            image_shape = (1, 28, 28)
        elif "cifar10" in dataset or "svhn" in dataset:
            image_shape = (3, 32, 32)
        else:
            raise ValueError(
                "dataset must be dirty-mnist, mnist-calib, mnist-expl or cifar10-c"
            )

        model_params = dict(
            model=model,  # ConvResNetSPN, ConvResNetDDU
            block="basic",  # basic, bottleneck
            layers=[2, 2, 2, 2],
            num_classes=10,
            image_shape=image_shape,
            einet_depth=5,
            einet_num_sums=20,
            einet_num_leaves=20,
            einet_num_repetitions=5,  # might be overwritten by optuna
            einet_leaf_type="Normal",
            einet_dropout=0.0,
            spec_norm_bound=0.95,  # only for ConvResNetSPN
            spectral_normalization=True,  # only for ConvResNetDDU
            mod=True,  # only for ConvResNetDDU
        )
        if loss == "discriminative" or loss == "noloss":
            train_params["lambda_v"] = 1.0
        elif loss == "generative":
            train_params["lambda_v"] = 0.0
        elif loss == "hybrid":
            train_params["lambda_v"] = 0.5
        elif loss == "hybrid_low":
            train_params["lambda_v"] = 0.1
        elif loss == "hybrid_very_low":
            train_params["lambda_v"] = 0.01
        elif loss == "hybrid_high":
            train_params["lambda_v"] = 0.9
        elif loss == "hybrid_very_high":
            train_params["lambda_v"] = 0.99
        else:
            raise ValueError(
                "loss must be discriminative, generative, hybrid, hybrid_low, hybrid_very_low or hybrid_high"
            )

        if training == "end-to-end":
            train_params["warmup_epochs"] = 0
            train_params["deactivate_backbone"] = False
        elif training == "seperate":
            train_params["warmup_epochs"] = 100
            train_params["deactivate_backbone"] = True
        elif training == "warmup":
            train_params["warmup_epochs"] = 100
            train_params["deactivate_backbone"] = False
        elif training == "backbone_only":
            train_params["warmup_epochs"] = 50
            train_params["deactivate_backbone"] = False
            train_params["num_epochs"] = 0
        elif training == "einet_only":
            train_params["warmup_epochs"] = 0
            train_params["deactivate_backbone"] = True
            train_params["num_epochs"] = 100
        else:
            raise ValueError(
                "training must be end-to-end, seperate, warmup, backbone_only or einet_only"
            )
        if pretrained_path is not None:
            train_params["pretrained_path"] = pretrained_path
        if training == "backbone_only":
            train_params, model_params = suggest_hps_backbone_only(
                trial, train_params, model_params
            )
        elif training == "einet_only":
            train_params, model_params = suggest_hps_einet_only(
                trial, train_params, model_params
            )
        else:
            train_params, model_params = suggest_hps(trial, train_params, model_params)
        if "mnist-calib" in dataset:
            try:
                return start_mnist_calib_run(
                    run_name, batch_sizes, model_params, train_params, trial
                )
            except Exception as e:
                if type(e) == optuna.exceptions.TrialPruned:
                    raise e
                print(e)
                traceback.print_exc()
                mlflow.set_tag("pruned", e)
                mlflow.end_run()
                raise optuna.TrialPruned()
        elif "mnist-expl" in dataset:
            model_params["explaining_vars"] = [0, 1, 2]  # rotations, cutoffs, noises
            train_params["highest_severity_train"] = 3
            # copy here, since some params are changed in the experiment
            try:
                val_loss_2 = start_mnist_expl_run(
                    run_name,
                    batch_sizes.copy(),
                    model_params.copy(),
                    train_params.copy(),
                    trial,
                )
                train_params["highest_severity_train"] = 4
                val_loss_4 = start_mnist_expl_run(
                    run_name, batch_sizes, model_params, train_params, trial
                )
                return (val_loss_2 + val_loss_4) / 2
            except Exception as e:
                if type(e) == optuna.exceptions.TrialPruned:
                    raise e
                print(e)
                traceback.print_exc()
                mlflow.set_tag("pruned", e)
                mlflow.end_run()
                raise optuna.TrialPruned()
        elif "dirty-mnist" in dataset:
            try:
                return start_dirty_mnist_run(
                    run_name, batch_sizes, model_params, train_params, trial
                )
            except Exception as e:
                if type(e) == optuna.exceptions.TrialPruned:
                    raise e
                print(e)
                traceback.print_exc()
                mlflow.set_tag("pruned", e)
                mlflow.end_run()
                raise optuna.TrialPruned()
        elif "cifar10-c-calib" in dataset:
            try:
                return start_cifar10_calib_run(
                    run_name, batch_sizes, model_params, train_params, trial
                )
            except Exception as e:
                if type(e) == optuna.exceptions.TrialPruned:
                    raise e
                print(e)
                traceback.print_exc()
                mlflow.set_tag("pruned", e)
                mlflow.end_run()
                raise optuna.TrialPruned()
        elif "cifar10-c-expl" in dataset:
            model_params["explaining_vars"] = list(range(19))
            train_params["corruption_levels_train"] = [0, 1]
            try:
                val_loss_2 = start_cifar10_expl_run(
                    run_name,
                    batch_sizes.copy(),
                    model_params.copy(),
                    train_params.copy(),
                    trial,
                )
                # train_params["corruption_levels_train"] = [0, 1, 2, 3]
                # val_loss_4 = start_cifar10_expl_run(
                #     run_name, batch_sizes, model_params, train_params, trial
                # )
                # return (val_loss_2 + val_loss_4) / 2
                return val_loss_2
            except Exception as e:
                if type(e) == optuna.exceptions.TrialPruned:
                    raise e
                print(e)
                traceback.print_exc()
                mlflow.set_tag("pruned", e)
                mlflow.end_run()
                raise optuna.TrialPruned()
        else:
            raise ValueError(
                "dataset must be mnist-calib, mnist-expl, dirty-mnist, cifar10-c or cifar10-c_expl"
            )

    exp = mlflow.get_experiment_by_name(dataset)
    runs = []
    if exp:
        query = f"attributes.run_name = '{run_name}'"
        runs = mlflow.search_runs([exp.experiment_id], query)
    n_trials = 15
    # trial_multiplier = 2 if "expl" in dataset else 1
    trial_multiplier = 1
    # only run experiment, if it wasnt run fully
    if len(runs) < n_trials * trial_multiplier:
        mlflow.set_experiment(dataset)
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=1,  # requires at least 3 results to start pruning
                n_warmup_steps=10,  # trial needs to log at least 10 steps before pruning
            ),
        )
        study.optimize(objective, n_trials=n_trials)


# Ziel vom Tuning: Testen, ob und wie gut Idee mit Einet funktioniert -> End-to-End training kann man spaeter herausfinden
# Nach Gespraech mit Christian neue Idee fuer schnelleres tuning:
# Erstes Tuning:
# - training schedule nur seperate
# - einmal backbones (DenseRes, ConvRes, ConvResDDU, Efficient) optimieren (Heute)
#    - dafuer nur LR (log) optimieren fuer jedes experiment (specnorm=0.95)
#    - maximal 50 epochs, 15 trials
#    - keine auswertung noetig
# - Ergebnis: 1 (twomoons) + 3 (mnist) + 3 (dirty) + 3 (cifar) = 10 models
# - Idealerweise bis morgen / heute abend fertig.
#
# Zweites Tuning:
# - Beste backbones raussuchen und accuracy berechnen und notieren(!)
# - Fuer jedes experiment:
#   - Lade passendes Backbone
#   - Nur seperate training (frozen backbone) (==ideal) -> end-to-end/warmup kann spaeter optimiert werden
#   - Optimiere einet_depth, einet_rep, einet_dropout (jeweils 2-3 werte), lr (log)
# - Alle losses, datasets und backbones (5 + 5 * 5 * 3) = 80 runs a 15 trials = 1200 trials

# AutoEncoder spaeter extra optimieren

# Alter tuning-versuch
# loss = ["discriminative", "generative", "hybrid_low", "hybrid_high", "hybrid", "hybrid_very_low"]
# training = ["end-to-end", "seperate", "warmup"]
# dataset = [
#     "two-moons",
#     "mnist-calib",
#     "mnist-expl",
#     "dirty-mnist",
#     "cifar10-c",
#     "cifar10-c_expl",
# ]
# models = [
#     "ConvResNetSPN",
#     "ConvResNetDDU",
#     "AutoEncoderSPN",
#     "EfficientNetSPN",
# ]

# for l in loss:
#     for t in training:
#         for d in dataset:
#             if d == "two-moons":
#                 tune_two_moons(l, t)
#             else:
#                 for m in models:
#                     tune_conv(d, l, t, m)

# Erstes Tuning
# dataset = [
#     "two-moons",
#     "mnist-calib",
#     # "mnist-expl",
#     "dirty-mnist",
#     "cifar10-c",
#     # "cifar10-c_expl",
# ]
# models = [
#     "ConvResNetSPN",
#     "ConvResNetDDU",
#     # "AutoEncoderSPN",
#     "EfficientNetSPN",
# ]
# for d in dataset:
#     if d == "two-moons":
#         tune_two_moons("discriminative", "backbone_only")
#         continue
#     for m in models:
#         tune_conv(d, "discriminative", "backbone_only", m)


def run_cifar_expl(dataset, model, loss, pretrained_path):
    mlflow.set_experiment(dataset)
    model_params = dict(
        model=model,
        block="basic",  # basic, bottleneck
        layers=[2, 2, 2, 2],
        num_classes=10,
        image_shape=(3, 32, 32),
        einet_depth=5,
        einet_num_sums=20,
        einet_num_leaves=20,
        einet_num_repetitions=5,
        einet_leaf_type="Normal",
        einet_dropout=0.0,
        spec_norm_bound=0.9,  # only for ConvResNetSPN
        spectral_normalization=True,  # only for ConvResNetDDU
        mod=True,  # only for ConvResNetDDU
    )
    if model == "ConvResNetSPN":
        lr = 0.002
    elif model == "ConvResNetDDU":
        lr = 0.02
    elif model == "EfficientNetSPN":
        lr = 0.015
    else:
        raise ValueError(
            "model must be ConvResNetSPN, ConvResNetDDU or EfficientNetSPN"
        )

    if loss == "discriminative":
        lambda_v = 1.0
    elif loss == "generative":
        lambda_v = 0.0
    elif loss == "hybrid":
        lambda_v = 0.5
    elif loss == "hybrid_low":
        lambda_v = 0.1
    elif loss == "hybrid_very_low":
        lambda_v = 0.01
    elif loss == "hybrid_high":
        lambda_v = 0.9
    elif loss == "hybrid_very_high":
        lambda_v = 0.99
    else:
        raise ValueError(
            "loss must be discriminative, generative, hybrid, hybrid_low, hybrid_very_low or hybrid_high"
        )
    train_params = dict(
        pretrained_path=pretrained_path,
        learning_rate_warmup=0.05,  # irrelevant
        learning_rate=lr,  # depends on model
        lambda_v=lambda_v,  # depends on loss
        warmup_epochs=0,
        num_epochs=100,
        deactivate_backbone=True,
        early_stop=10,
    )
    batch_sizes = dict(resnet=512)
    model_params["explaining_vars"] = list(range(19))
    train_params["corruption_levels_train"] = [0, 1]
    training = "einet_only"
    run_name = f"{loss}_{training}_{model}_manual"
    try:
        val_loss_2 = start_cifar10_expl_run(
            run_name,
            batch_sizes.copy(),
            model_params.copy(),
            train_params.copy(),
            trial=None,
        )
        train_params["corruption_levels_train"] = [0, 1, 2, 3]
        val_loss_4 = start_cifar10_expl_run(
            run_name, batch_sizes, model_params, train_params, trial=None
        )
        return (val_loss_2 + val_loss_4) / 2
    except Exception as e:
        print(e)
        traceback.print_exc()
        mlflow.set_tag("pruned", e)
        mlflow.end_run()


def run_mnist_expl(dataset, model, loss, pretrained_path):
    mlflow.set_experiment(dataset)
    model_params = dict(
        model=model,
        block="basic",  # basic, bottleneck
        layers=[2, 2, 2, 2],
        num_classes=10,
        image_shape=(1, 28, 28),
        einet_depth=5,
        einet_num_sums=20,
        einet_num_leaves=20,
        einet_num_repetitions=5,
        einet_leaf_type="Normal",
        einet_dropout=0.0,
        spec_norm_bound=0.9,  # only for ConvResNetSPN
        spectral_normalization=True,  # only for ConvResNetDDU
        mod=True,  # only for ConvResNetDDU
    )
    if model == "ConvResNetSPN":
        lr = 0.002
    elif model == "ConvResNetDDU":
        lr = 0.02
    elif model == "EfficientNetSPN":
        lr = 0.015
    else:
        raise ValueError(
            "model must be ConvResNetSPN, ConvResNetDDU or EfficientNetSPN"
        )

    if loss == "discriminative":
        lambda_v = 1.0
    elif loss == "generative":
        lambda_v = 0.0
    elif loss == "hybrid":
        lambda_v = 0.5
    elif loss == "hybrid_low":
        lambda_v = 0.1
    elif loss == "hybrid_very_low":
        lambda_v = 0.01
    elif loss == "hybrid_high":
        lambda_v = 0.9
    elif loss == "hybrid_very_high":
        lambda_v = 0.99
    else:
        raise ValueError(
            "loss must be discriminative, generative, hybrid, hybrid_low, hybrid_very_low or hybrid_high"
        )
    train_params = dict(
        pretrained_path=pretrained_path,
        learning_rate_warmup=0.05,  # irrelevant
        learning_rate=lr,  # depends on model
        lambda_v=lambda_v,  # depends on loss
        warmup_epochs=0,
        num_epochs=100,
        # num_epochs=1,
        deactivate_backbone=True,
        early_stop=10,
    )
    batch_sizes = dict(resnet=512)
    model_params["explaining_vars"] = [0, 1, 2]  # rotations, cutoffs, noises
    train_params["highest_severity_train"] = 2
    training = "einet_only"
    run_name = f"{loss}_{training}_{model}_mspn"
    try:
        val_loss_2 = start_mnist_expl_run(
            run_name,
            batch_sizes.copy(),
            model_params.copy(),
            train_params.copy(),
            trial=None,
        )
        # train_params["highest_severity_train"] = 4
        # val_loss_4 = start_mnist_expl_run(
        #     run_name, batch_sizes, model_params, train_params, trial=None
        # )
        # return (val_loss_2 + val_loss_4) / 2
        return val_loss_2
    except Exception as e:
        print(e)
        traceback.print_exc()
        mlflow.set_tag("pruned", e)
        mlflow.end_run()


def run_dense_resnet(dataset, loss, training, model, pretrained_path=None):
    mlflow.set_experiment(dataset)
    run_name = f"{loss}_{training}_{model}"

    batch_sizes = dict(resnet=512)
    if "two-moons" in dataset:
        num_classes = 2
    elif "figure6" in dataset:
        num_classes = 3
    else:
        num_classes = 10
    if "SPN" in model:
        leaf_type = RatNormal
        leaf_kwargs = {
            "min_sigma": 0.000001,
            "max_sigma": 5.0,  # good for twomoons
            # "max_sigma": 50.0,  # good for figure6
            # "max_sigma": 20.0,
        }
        if "figure6" in dataset:
            # lambda should be 0.3 or 0.5
            einet_depth = 3
            einet_num_sums = 10
            einet_num_leaves = 15
            einet_num_repetitions = 11
        elif "two-moons" in dataset:
            # (also works for figure6 with dropout=0.01, but occams razor: smaller model is better)
            # lambda should be 0.5 or 0.7
            einet_depth = 5
            einet_num_sums = 10
            einet_num_leaves = 15
            einet_num_repetitions = 11
        else:
            raise NotImplementedError()

        model_params = dict(
            model=model,
            num_classes=num_classes,
            input_dim=2,
            num_layers=3,
            num_hidden=32,
            spec_norm_bound=0.95,
            einet_depth=einet_depth,
            einet_num_sums=einet_num_sums,
            einet_num_leaves=einet_num_leaves,
            einet_num_repetitions=einet_num_repetitions,
            einet_leaf_type=leaf_type,
            einet_leaf_kwargs=leaf_kwargs,
            einet_dropout=0.00,
        )
    elif "SNGP" in model:
        model_params = dict(
            model=model,
            train_num_data=1000 if dataset == "two-moons" else 3300,
            train_batch_size=batch_sizes["resnet"],
            num_classes=num_classes,
            input_dim=2,
            num_layers=3,
            num_hidden=32,
            # num_hidden=640,
            spec_norm_bound=0.95,
        )
    elif "GMM" in model:
        model_params = dict(
            model=model,
            num_classes=num_classes,
            input_dim=2,
            num_layers=4,
            # num_hidden=32,
            num_hidden=128,
            # spec_norm_bound=0.95,
            spec_norm_bound=0.65,
        )
    train_params = dict(
        early_stop=100,
        learning_rate_warmup=0.03,
        # learning_rate=0.025,
        # learning_rate=0.03,  # good for sngp
        # learning_rate=0.09,  # good for spn
        learning_rate=0.03,  # good for both
    )
    if loss == "discriminative" or loss == "noloss":
        train_params["lambda_v"] = 1.0
    elif loss == "generative":
        train_params["lambda_v"] = 0.0
    elif loss == "hybrid":
        train_params["lambda_v"] = 0.5
    elif loss == "hybrid_low":
        train_params["lambda_v"] = 0.1
    elif loss == "hybrid_mid_low":
        train_params["lambda_v"] = 0.3
    elif loss == "hybrid_mid_high":
        train_params["lambda_v"] = 0.7
    elif loss == "hybrid_high":
        train_params["lambda_v"] = 0.9
    else:
        raise ValueError(
            "loss must be discriminative, generative, hybrid, hybrid_low, hybrid_very_low or hybrid_high"
        )

    if training == "end-to-end":
        train_params["warmup_epochs"] = 0
        train_params["deactivate_backbone"] = False
        train_params["num_epochs"] = 400
    elif training == "seperate":
        train_params["warmup_epochs"] = 400
        train_params["deactivate_backbone"] = True
        train_params["num_epochs"] = 400
    elif training == "warmup":
        train_params["warmup_epochs"] = 100
        train_params["deactivate_backbone"] = False
        train_params["num_epochs"] = 300
    elif training == "backbone_only":
        train_params["warmup_epochs"] = 400
        train_params["deactivate_backbone"] = False
        train_params["num_epochs"] = 0
    elif training == "einet_only":
        train_params["warmup_epochs"] = 0
        train_params["deactivate_backbone"] = True
        train_params["num_epochs"] = 400
    else:
        raise ValueError(
            "training must be end-to-end, seperate, warmup, backbone_only or einet_only"
        )
    if pretrained_path is not None:
        train_params["pretrained_path"] = pretrained_path
    if "two-moons" in dataset:
        try:
            start_two_moons_run(
                run_name,
                batch_sizes,
                model_params,
                train_params,
                None,
            )
        except Exception as e:
            print(e)
            traceback.print_exc()
            mlflow.set_tag("pruned", e)
            mlflow.end_run()
    elif "latent_figure6" in dataset:
        try:
            start_latent_figure6_run(
                run_name,
                batch_sizes,
                model_params,
                train_params,
                None,
            )
        except Exception as e:
            print(e)
            traceback.print_exc()
            mlflow.set_tag("pruned", e)
            mlflow.end_run()
    elif "figure6" in dataset:
        try:
            start_figure6_run(
                run_name,
                batch_sizes,
                model_params,
                train_params,
                None,
            )
        except Exception as e:
            print(e)
            traceback.print_exc()
            mlflow.set_tag("pruned", e)
            mlflow.end_run()


# Zweites Tuning
loss = [
    "hybrid",
    # "hybrid_mid_low",
    # "hybrid_mid_high",
    # "hybrid_high",
    # "hybrid_low",
    # "generative",
    # "discriminative",
]
dataset = [
    # "two-moons",
    # "figure6",
    # "latent_figure6",
    # "dirty-mnist",
    # "mnist-calib",
    # "mnist-expl",
    "cifar10-c-calib",
    "svhn-c-calib",
    # "cifar10-expl-bright",
    # "cifar10-c-expl",
    # "svhn-c-expl",
]
dense_models = [
    # "DenseResNetSPN",
    # "DenseResNetGMM",
    "DenseResNetSNGP",
]
models = [
    # "EfficientNetSPN",
    # "ConvResNetSPN",
    # "ConvResNetDDU",
    # "EfficientNetGMM",
    # "ConvResNetDDUGMM",
    "EfficientNetDropout",
    # "EfficientNetEnsemble",
    # "EfficientNetSNGP",
]
pretrained_backbones = {
    # acc: 1
    "two-moons": "814351535813234998/48fe608aa28642968dfcaad0201a47e0/artifacts/model",
    "figure6": "155846836899219289/3772e86556b449fe9c763d42e87a69bd/artifacts/model",
    "mnist-calib": {
        # val-acc: 0.9919
        "ConvResNetSPN": "175904093117473539/196a38010d6846cdacf663229314882b/artifacts/model",
        # val-acc: 0.9953
        "ConvResNetDDU": "175904093117473539/2b6023045bce4529bd3b49a4d3313e08/artifacts/model",
        "ConvResNetDDUGMM": "175904093117473539/2b6023045bce4529bd3b49a4d3313e08/artifacts/model",
        # val-acc: 0.9918
        "EfficientNetSPN": "175904093117473539/75ec0d48354845278b00fc8aec0e68f9/artifacts/model",
        # val-acc: 0.9918
        "EfficientNetGMM": "175904093117473539/75ec0d48354845278b00fc8aec0e68f9/artifacts/model",
    },
    # same as calib
    "mnist-expl": {
        "ConvResNetSPN": "175904093117473539/196a38010d6846cdacf663229314882b/artifacts/model",
        "ConvResNetDDU": "175904093117473539/2b6023045bce4529bd3b49a4d3313e08/artifacts/model",
        "ConvResNetDDUGMM": "175904093117473539/2b6023045bce4529bd3b49a4d3313e08/artifacts/model",
        # "EfficientNetSPN": "175904093117473539/75ec0d48354845278b00fc8aec0e68f9/artifacts/model",
        "EfficientNetSPN": "987712940205555914/79e5f60ba8644d4ebbaf0faccce19010/artifacts/model",
        "EfficientNetGMM": "175904093117473539/75ec0d48354845278b00fc8aec0e68f9/artifacts/model",
    },
    "dirty-mnist": {
        # val-acc: 0.8954
        "ConvResNetSPN": "958692786192727381/1cd3bd7d4f974190a078c7e3c3362cb0/artifacts/model",
        # val-acc: 0.8973
        "ConvResNetDDU": "958692786192727381/6b5c21c4f3da4506b85bb63cd9683e39/artifacts/model",
        "ConvResNetDDUGMM": "958692786192727381/6b5c21c4f3da4506b85bb63cd9683e39/artifacts/model",
        # val-acc: 0.8944
        "EfficientNetSPN": "958692786192727381/54bb623d51ac41b0b4c19717c39b75d1/artifacts/model",
        "EfficientNetGMM": "958692786192727381/54bb623d51ac41b0b4c19717c39b75d1/artifacts/model",
    },
    "cifar10-c-calib": {
        # val-acc: 0.8188
        "ConvResNetSPN": "344247532890804598/17bdc2e7a26c4f529ce41483842362e0/artifacts/model",
        # val-acc: 0.8850
        "ConvResNetDDU": "344247532890804598/725e12384abc4a05848e88bff062c5ef/artifacts/model",
        "ConvResNetDDUGMM": "344247532890804598/725e12384abc4a05848e88bff062c5ef/artifacts/model",
        # val-acc: 0.8496
        "EfficientNetSPN": "279248034225110540/f3752cea01f7472aade2e60dbf8fedcb/artifacts/model",
        "EfficientNetGMM": "344247532890804598/c0ebabeb76914132a1d162bb068389db/artifacts/model",
    },
    # same as calib
    "cifar10-c-expl": {
        "ConvResNetSPN": "344247532890804598/17bdc2e7a26c4f529ce41483842362e0/artifacts/model",
        "ConvResNetDDU": "344247532890804598/725e12384abc4a05848e88bff062c5ef/artifacts/model",
        "EfficientNetSPN": "298139550611321154/1032be5fb5304f58ab1564ffd819ff3c/artifacts/model",
    },
    "svhn-c-expl": {
        "EfficientNetSPN": "354955436886369284/d0b7ff2869cf47cabee81603af9c273a/artifacts/model"
    },
    "svhn-c-calib": {
        "EfficientNetSPN": "382722780317026903/e906f4cb0e3640fb87fb37dfc907944f/artifacts/model"
    },
}

trained_models = {
    "mnist-calib": {
        # "EfficientNetSPN": "987712940205555914/b5320090ae7d4dd7a971f29207ac8097/artifacts/model",
        "EfficientNetSPN": "640832573776399546/7f8def9a908b4b008038f1f1be9f87bf/artifacts/model"
    },
    "mnist-expl": {
        # "EfficientNetSPN": "764598691207333922/02852d15bd2446f5bfc021b5043f2a29/artifacts/model",
        # "EfficientNetSPN": "764598691207333922/9e133cfbafbd4df2acac15464349f1ec/artifacts/model"
        # "EfficientNetSPN": "987712940205555914/eac48c7b7189427e904a604fb7772294/artifacts/model",
        "EfficientNetSPN": "987712940205555914/1e17e9a931aa4217bf844d83e5f81c0f/artifacts/model"
    },
    "cifar10-c-expl": {
        "EfficientNetSPN": "298139550611321154/240cf2315d2945bdad7658ae4a65cb83/artifacts/model"
    },
    "cifar10-c-calib": {
        "EfficientNetSPN": "279248034225110540/f3752cea01f7472aade2e60dbf8fedcb/artifacts/model",
    },
    "dirty-mnist": {
        "hybrid": {
            "EfficientNetSPN": "627860869402028747/9765a8e3d04a4b66b0ba854dbb9c03a4/artifacts/model",
            "ConvResNetDDU": "627860869402028747/a343319ac4db4a388f1dfef99fdf1789/artifacts/model",
        },
        "hybrid_low": {
            "ConvResNetDDU": "627860869402028747/becd4bd9a3824ebbbe09d9ce2862b6d1/artifacts/model",
            "EfficientNetSPN": "627860869402028747/85ab40cb132640828b499f3008cd4415/artifacts/model",
        },
    },
    "svhn-c-calib": {
        "EfficientNetSPN": "382722780317026903/e906f4cb0e3640fb87fb37dfc907944f/artifacts/model"
    },
    "svhn-c-expl": {
        "EfficientNetSPN": "771929259740930885/77863a62033042b999743a3434edc708/artifacts/model"
    },
}

# pretrained_path = pretrained_backbones["two-moons"]
# pretrained_path = "/data_docker/mlartifacts/" + pretrained_path + "/state_dict.pth"
# run_two_moons("hybrid_high", "einet_only", pretrained_path)

for d in dataset:
    if d == "two-moons" or "figure6" in d:
        for m in dense_models:
            # pretrained_path = None
            # run_dense_resnet(d, l, "seperate", m, pretrained_path)
            # run_dense_resnet(d, l, "einet_only", m, pretrained_path)
            if "SNGP" in m:
                l = "discriminative"
                run_dense_resnet(d, l, "end-to-end", m, pretrained_path=None)
                continue

            # pretrained_path = pretrained_backbones[d]
            # pretrained_path = (
            #     "/data_docker/mlartifacts/" + pretrained_path + "/state_dict.pth"
            # )
            pretrained_path = None
            if "GMM" in m:
                l = "discriminative"
                run_dense_resnet(d, l, "backbone_only", m, pretrained_path)
            elif "SPN" in m:
                for l in loss:
                    # run_dense_resnet(d, l, "einet_only", m, pretrained_path)
                    run_dense_resnet(d, l, "seperate", m, pretrained_path)
            continue
        continue
    for m in models:
        if "SNGP" in m:
            l = "discriminative"
            run_conv(d, l, "end-to-end", m, pretrained_path=None)
            continue
        elif "GMM" in m or "Dropout" in m or "Ensemble" in m:
            l = "discriminative"
            run_conv(d, l, "backbone_only", m, pretrained_path=None)
            continue
        elif "SPN" in m:
            for l in loss:
                # pretrained_path = pretrained_backbones[d][m]
                # # pretrained_path = trained_models[d][m]
                # pretrained_path = (
                #     "/data_docker/mlartifacts/" + pretrained_path + "/state_dict.pth"
                # )
                # pretrained_path = None
                run_conv(d, l, "seperate", m, pretrained_path=None)
                # run_conv(d, l, "backbone_only", m, pretrained_path=None)
                # run_conv(d, l, "einet_only", m, pretrained_path)
            continue
        # pretrained_path = pretrained_backbones[d][m]
        # pretrained_path = trained_models[d][m]
        # pretrained_path = trained_models[d][l][m]
        # pretrained_path = (
        #     "/data_docker/mlartifacts/" + pretrained_path + "/state_dict.pth"
        # )
        # pretrained_path = None
        # tune_conv(d, l, "end-to-end", m, pretrained_path)
        # run_conv(d, l, "einet_only", m, pretrained_path)
        # run_conv(d, l, "eval_only", m, pretrained_path)
        # run_conv(d, l, "seperate", m, pretrained_path)
        # run_conv(dataset, l, "seperate", m)


# path = (
#     "/data_docker/mlartifacts/"
#     + trained_models["mnist-expl"]["EfficientNetSPN"]
#     + "/state_dict.pth"
# )
# model_params = dict(
#     model="EfficientNetSPN",  # ConvResNetSPN, ConvResNetDDU
#     num_classes=10,
#     image_shape=(1, 28, 28),
#     explaining_vars=[0, 1, 2],
#     einet_depth=5,
#     einet_num_sums=20,
#     einet_num_leaves=20,
#     einet_num_repetitions=5,
#     einet_leaf_type="Normal",
#     einet_dropout=0.0,
# )
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# mnist_expl_manual_evaluation(model_params, path, device)

# path = (
#     "/data_docker/mlartifacts/"
#     + pretrained_backbones["mnist-expl"]["EfficientNetSPN"]
#     + "/state_dict.pth"
# )
# run_mnist_expl("mnist-expl-new", "EfficientNetSPN", "hybrid_very_low", path)

# path = (
#     "/data_docker/mlartifacts/"
#     + pretrained_backbones["cifar10-c-expl"]["EfficientNetSPN"]
#     + "/state_dict.pth"
# )
# run_cifar_expl("cifar10-c-expl-new", "EfficientNetSPN", "hybrid_very_low", path)
