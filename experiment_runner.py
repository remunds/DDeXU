import traceback
import torch
import mlflow
import optuna
from two_moons_experiment import start_two_moons_run
from mnist_calib_experiment import start_mnist_calib_run
from mnist_expl_experiment import start_mnist_expl_run, mnist_expl_manual_evaluation
from dirty_mnist_experiment import start_dirty_mnist_run
from cifar10_expl_experiment import start_cifar10_expl_run
from cifar10_calib_experiment import start_cifar10_calib_run

torch.manual_seed(0)
# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")


def suggest_hps(trial, train_params, model_params):
    einet_depth = trial.suggest_categorical("einet_depth", [3, 5])
    einet_rep = trial.suggest_categorical("einet_rep", [1, 3])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    schedule_step_size = trial.suggest_int("schedule_step_size", 5, 20)
    schedule_gamma = trial.suggest_float("schedule_gamma", 0.5, 0.9)
    if train_params["warmup_epochs"] > 0:
        lr_warmup = trial.suggest_float("lr_warmup", 1e-5, 1e-1, log=True)
        schedule_step_size_warmup = trial.suggest_int(
            "schedule_step_size_warmup", 5, 20
        )
        schedule_gamma_warmup = trial.suggest_float("schedule_gamma_warmup", 0.5, 0.9)
        train_params["lr_schedule_warmup_step_size"] = schedule_step_size_warmup
        train_params["lr_schedule_warmup_gamma"] = schedule_gamma_warmup
        train_params["learning_rate_warmup"] = lr_warmup
    else:
        train_params["learning_rate_warmup"] = 0
        train_params["lr_schedule_warmup_step_size"] = 0
        train_params["lr_schedule_warmup_gamma"] = 0

    train_params["learning_rate"] = lr
    train_params["lr_schedule_step_size"] = schedule_step_size
    train_params["lr_schedule_gamma"] = schedule_gamma
    model_params["einet_depth"] = einet_depth
    model_params["einet_num_repetitions"] = einet_rep

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
    elif "cifar10" in dataset:
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
        einet_depth=5,  # might be overwritten by optuna
        einet_num_sums=20,
        einet_num_leaves=20,
        einet_num_repetitions=5,  # might be overwritten by optuna
        einet_leaf_type="Normal",
        einet_dropout=0.0,
        spec_norm_bound=0.95,  # only for ConvResNetSPN
        spectral_normalization=True,  # only for ConvResNetDDU
        mod=True,  # only for ConvResNetDDU
    )
    train_params = dict(
        pretrained_path=pretrained_path,
        learning_rate_warmup=0.05,  # irrelevant
        num_epochs=100,
        early_stop=5,
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
        train_params["num_epochs"] = 10
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
    elif model == "ConvResNetDDU":
        lr = 0.02
    elif model == "EfficientNetSPN":
        lr = 0.015
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
        train_params["highest_severity_train"] = 2
        # copy here, since some params are changed in the experiment
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
            warmup_epochs=100,
            num_epochs=100,
            early_stop=5,
        )
        if "mnist" in dataset:
            image_shape = (1, 28, 28)
        elif "cifar10" in dataset:
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
            einet_depth=3,  # might be overwritten by optuna
            einet_num_sums=20,
            einet_num_leaves=20,
            einet_num_repetitions=1,  # might be overwritten by optuna
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
            train_params["highest_severity_train"] = 2
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
                train_params["corruption_levels_train"] = [0, 1, 2, 3]
                val_loss_4 = start_cifar10_expl_run(
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
    trial_multiplier = 2 if "expl" in dataset else 1
    # only run experiment, if it wasnt run fully
    if len(runs) < n_trials * trial_multiplier:
        mlflow.set_experiment(dataset)
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=3,  # requires at least 3 results to start pruning
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
    run_name = f"{loss}_{training}_{model}_manual"
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


def run_two_moons(dataset, loss, training, pretrained_path=None):
    print("New run of two moons")

    # only run experiment, if it wasnt run fully
    mlflow.set_experiment(dataset)
    run_name = f"{loss}_{training}"

    batch_sizes = dict(resnet=512)
    model_params_dense = dict(
        num_classes=2,
        input_dim=2,
        num_layers=3,
        num_hidden=32,
        spec_norm_bound=0.95,
        einet_depth=5,
        einet_num_sums=20,
        einet_num_leaves=20,
        einet_num_repetitions=5,
        einet_leaf_type="Normal",
        einet_dropout=0.0,
    )
    train_params = dict(
        num_epochs=100,
        early_stop=10,
        learning_rate_warmup=0.05,
        learning_rate=0.09,
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
        # train_params["lambda_v"] = 0.95
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
    try:
        start_two_moons_run(
            run_name,
            batch_sizes,
            model_params_dense,
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
    # "hybrid",
    # "hybrid_low",
    # "generative",
    # "discriminative",
    "hybrid_high",
    # "hybrid_very_high",
    # "hybrid_very_low",
]
dataset = [
    # "two-moons",
    # "dirty-mnist",
    "mnist-calib",
    # "mnist-expl",
    # "cifar10-c-calib",
    # "cifar10-c-expl",
]
models = [
    # "ConvResNetSPN",
    # "ConvResNetDDU",
    "EfficientNetSPN",
]
pretrained_backbones = {
    # acc: 1
    "two-moons": "814351535813234998/48fe608aa28642968dfcaad0201a47e0/artifacts/model",
    "mnist-calib": {
        # val-acc: 0.9919
        "ConvResNetSPN": "175904093117473539/196a38010d6846cdacf663229314882b/artifacts/model",
        # val-acc: 0.9953
        "ConvResNetDDU": "175904093117473539/2b6023045bce4529bd3b49a4d3313e08/artifacts/model",
        # val-acc: 0.9918
        "EfficientNetSPN": "175904093117473539/75ec0d48354845278b00fc8aec0e68f9/artifacts/model",
    },
    # same as calib
    "mnist-expl": {
        "ConvResNetSPN": "175904093117473539/196a38010d6846cdacf663229314882b/artifacts/model",
        "ConvResNetDDU": "175904093117473539/2b6023045bce4529bd3b49a4d3313e08/artifacts/model",
        "EfficientNetSPN": "175904093117473539/75ec0d48354845278b00fc8aec0e68f9/artifacts/model",
    },
    "dirty-mnist": {
        # val-acc: 0.8954
        "ConvResNetSPN": "958692786192727381/1cd3bd7d4f974190a078c7e3c3362cb0/artifacts/model",
        # val-acc: 0.8973
        "ConvResNetDDU": "958692786192727381/6b5c21c4f3da4506b85bb63cd9683e39/artifacts/model",
        # val-acc: 0.8944
        "EfficientNetSPN": "958692786192727381/54bb623d51ac41b0b4c19717c39b75d1/artifacts/model",
    },
    "cifar10-c-calib": {
        # val-acc: 0.8188
        "ConvResNetSPN": "344247532890804598/17bdc2e7a26c4f529ce41483842362e0/artifacts/model",
        # val-acc: 0.8850
        "ConvResNetDDU": "344247532890804598/725e12384abc4a05848e88bff062c5ef/artifacts/model",
        # val-acc: 0.8496
        "EfficientNetSPN": "344247532890804598/c0ebabeb76914132a1d162bb068389db/artifacts/model",
    },
    # same as calib
    "cifar10-c-expl": {
        "ConvResNetSPN": "344247532890804598/17bdc2e7a26c4f529ce41483842362e0/artifacts/model",
        "ConvResNetDDU": "344247532890804598/725e12384abc4a05848e88bff062c5ef/artifacts/model",
        "EfficientNetSPN": "344247532890804598/c0ebabeb76914132a1d162bb068389db/artifacts/model",
    },
}

trained_models = {
    "mnist-calib": {
        # "EfficientNetSPN": "987712940205555914/b5320090ae7d4dd7a971f29207ac8097/artifacts/model",
        "EfficientNetSPN": "640832573776399546/7f8def9a908b4b008038f1f1be9f87bf/artifacts/model"
    },
    "mnist-expl": {
        # "EfficientNetSPN": "764598691207333922/02852d15bd2446f5bfc021b5043f2a29/artifacts/model",
        "EfficientNetSPN": "764598691207333922/9e133cfbafbd4df2acac15464349f1ec/artifacts/model"
    },
    "cifar10-c-expl": {
        "EfficientNetSPN": "718553087440563724/d7f46d12439e4ac48a4284303ee92d40/artifacts/model",
    },
}

# pretrained_path = pretrained_backbones["two-moons"]
# pretrained_path = "/data_docker/mlartifacts/" + pretrained_path + "/state_dict.pth"
# run_two_moons("hybrid_high", "einet_only", pretrained_path)

for d in dataset:
    for l in loss:
        if d == "two-moons":
            pretrained_path = pretrained_backbones[d]
            pretrained_path = (
                "/data_docker/mlartifacts/" + pretrained_path + "/state_dict.pth"
            )
            dataset = d + "-newLeafKwargs"
            run_two_moons(dataset, l, "einet_only", pretrained_path)
            continue
        for m in models:
            # pretrained_path = pretrained_backbones[d][m]
            pretrained_path = trained_models[d][m]
            pretrained_path = (
                "/data_docker/mlartifacts/" + pretrained_path + "/state_dict.pth"
            )
            dataset = d + "-newLeafKwargs"
            run_conv(dataset, l, "eval_only", m, pretrained_path)
            # run_conv(dataset, l, "einet_only", m, pretrained_path)
            # run_cifar_expl(d, m, l, pretrained_path)


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
