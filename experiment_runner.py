import traceback
import torch
import mlflow
import optuna
from two_moons_experiment import start_two_moons_run
from mnist_calib_experiment import start_mnist_calib_run
from mnist_expl_experiment import start_mnist_expl_run
from dirty_mnist_experiment import start_dirty_mnist_run
from cifar10_expl_experiment import start_cifar10_expl_run
from cifar10_calib_experiment import start_cifar10_calib_run

torch.manual_seed(0)
# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# model_params_conv = dict(
#     model="ConvResNetSPN",  # ConvResNetSPN, ConvResNetDDU
#     block="basic",  # basic, bottleneck
#     layers=[2, 2, 2, 2],
#     num_classes=10,
#     image_shape=(1, 28, 28),
#     einet_depth=3,
#     einet_num_sums=20,
#     einet_num_leaves=20,
#     einet_num_repetitions=1,
#     einet_leaf_type="Normal",
#     einet_dropout=0.0,
#     spec_norm_bound=0.9,  # only for ConvResNetSPN
#     spectral_normalization=True,  # only for ConvResNetDDU
#     mod=True,  # only for ConvResNetDDU
# )
# train_params = dict(
#     learning_rate_warmup=0.05,
#     learning_rate=0.05,
#     lambda_v=0.995,
#     warmup_epochs=100,
#     num_epochs=100,
#     deactivate_backbone=True,
#     lr_schedule_warmup_step_size=10,
#     lr_schedule_warmup_gamma=0.5,
#     lr_schedule_step_size=10,
#     lr_schedule_gamma=0.5,
#     early_stop=10,
# )


# TODO: in the long run, it would be nice to optimize all HP's for accuracy and ECE
# Manually select the following setups:
# - fully discriminative, fully generative, hybrid loss
# - for end-to-end, seperate and warmup training
# this is already many different runs for each dataset
# Then, tune following hyperparameters for each of these runs:
# einet_depth (3 and 5), einet_num_repetitions, lr, lr_warmup, schedule_step_size, schedule_gamma
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


def tune_two_moons(loss, training):
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
            spec_norm_bound=0.9,
            # einet_depth=3,
            einet_num_sums=20,
            einet_num_leaves=20,
            # einet_num_repetitions=1,
            einet_leaf_type="Normal",
            einet_dropout=0.0,
        )
        if loss == "discriminative":
            train_params["lambda_v"] = 1.0
        elif loss == "generative":
            train_params["lambda_v"] = 0.0
        elif loss == "hybrid":
            train_params["lambda_v"] = 0.5
        elif loss == "hybrid_low":
            train_params["lambda_v"] = 0.1
        elif loss == "hybrid_high":
            train_params["lambda_v"] = 0.9
        else:
            raise ValueError("loss must be discriminative, generative or hybrid")

        if training == "end-to-end":
            train_params["warmup_epochs"] = 0
            train_params["deactivate_backbone"] = False
        elif training == "seperate":
            train_params["warmup_epochs"] = 100
            train_params["deactivate_backbone"] = True
        elif training == "warmup":
            train_params["warmup_epochs"] = 100
            train_params["deactivate_backbone"] = False
        else:
            raise ValueError("training must be end-to-end, seperate or warmup")

        train_params, model_params_dense = suggest_hps(
            trial, train_params, model_params_dense
        )
        try:
            return start_two_moons_run(
                run_name, batch_sizes, model_params_dense, train_params, trial
            )
        except Exception as e:
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
    n_trials = 10
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


def tune_conv(dataset, loss, training, model):
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
            # einet_depth=3,
            einet_num_sums=20,
            einet_num_leaves=20,
            # einet_num_repetitions=1,
            einet_leaf_type="Normal",
            einet_dropout=0.0,
            spec_norm_bound=0.9,  # only for ConvResNetSPN
            spectral_normalization=True,  # only for ConvResNetDDU
            mod=True,  # only for ConvResNetDDU
        )
        if loss == "discriminative":
            train_params["lambda_v"] = 1.0
        elif loss == "generative":
            train_params["lambda_v"] = 0.0
        elif loss == "hybrid":
            train_params["lambda_v"] = 0.5
        elif loss == "hybrid_low":
            train_params["lambda_v"] = 0.1
        elif loss == "hybrid_high":
            train_params["lambda_v"] = 0.9
        else:
            raise ValueError("loss must be discriminative, generative or hybrid")

        if training == "end-to-end":
            train_params["warmup_epochs"] = 0
            train_params["deactivate_backbone"] = False
        elif training == "seperate":
            train_params["warmup_epochs"] = 100
            train_params["deactivate_backbone"] = True
        elif training == "warmup":
            train_params["warmup_epochs"] = 100
            train_params["deactivate_backbone"] = False
        else:
            raise ValueError("training must be end-to-end, seperate or warmup")

        train_params, model_params = suggest_hps(trial, train_params, model_params)
        if dataset == "mnist-calib":
            try:
                return start_mnist_calib_run(
                    run_name, batch_sizes, model_params, train_params, trial
                )
            except Exception as e:
                print(e)
                traceback.print_exc()
                mlflow.set_tag("pruned", e)
                mlflow.end_run()
                raise optuna.TrialPruned()
        elif dataset == "mnist-expl":
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
                print(e)
                traceback.print_exc()
                mlflow.set_tag("pruned", e)
                mlflow.end_run()
                raise optuna.TrialPruned()
        elif dataset == "dirty-mnist":
            try:
                return start_dirty_mnist_run(
                    run_name, batch_sizes, model_params, train_params, trial
                )
            except Exception as e:
                print(e)
                traceback.print_exc()
                mlflow.set_tag("pruned", e)
                mlflow.end_run()
                raise optuna.TrialPruned()
        elif dataset == "cifar10-c":
            try:
                return start_cifar10_calib_run(
                    run_name, batch_sizes, model_params, train_params, trial
                )
            except Exception as e:
                print(e)
                traceback.print_exc()
                mlflow.set_tag("pruned", e)
                mlflow.end_run()
                raise optuna.TrialPruned()
        elif dataset == "cifar10-c_expl":
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
    n_trials = 10
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


loss = ["discriminative", "generative", "hybrid_low", "hybrid_high", "hybrid"]
training = ["end-to-end", "seperate", "warmup"]
dataset = [
    "two-moons",
    "mnist-calib",
    "mnist-expl",
    "dirty-mnist",
    "cifar10-c",
    "cifar10-c_expl",
]

# for l in loss:
#     for t in training:
#         for d in dataset:
#             if d == "two-moons":
#                 tune_two_moons(l, t)
#             else:
#                 tune_conv(d, l, t, "ConvResNetSPN")
#                 tune_conv(d, l, t, "ConvResNetDDU")

tune_conv("cifar10-c_expl", "hybrid", "end-to-end", "AutoEncoderSPN")
