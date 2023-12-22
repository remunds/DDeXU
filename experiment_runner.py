import torch
import mlflow
from two_moons_experiment import start_two_moons_run
from mnist_calib_experiment import start_mnist_calib_run
from mnist_expl_experiment import start_mnist_expl_run
from dirty_mnist_experiment import start_dirty_mnist_run
from cifar10_expl_experiment import start_cifar10_expl_run
from cifar10_calib_experiment import start_cifar10_calib_run

torch.manual_seed(0)
# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

batch_sizes = dict(resnet=512)
model_params_dense = dict(
    input_dim=2,
    output_dim=2,
    num_layers=3,
    num_hidden=32,
    spec_norm_bound=0.9,
    einet_depth=3,
    einet_num_sums=20,
    einet_num_leaves=20,
    einet_num_repetitions=1,
    einet_leaf_type="Normal",
    einet_dropout=0.0,
)
model_params_conv = dict(
    model="ConvResNetSPN",  # ConvResNetSPN, ConvResNetDDU
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
    warmup_epochs=1,
    num_epochs=1,
    deactivate_resnet=True,
    lr_schedule_warmup_step_size=10,
    lr_schedule_warmup_gamma=0.5,
    lr_schedule_step_size=10,
    lr_schedule_gamma=0.5,
    early_stop=10,
)
run_name = "end_to_end"

# mlflow.set_experiment("two-moons")
# start_two_moons_run(run_name, batch_sizes, model_params_dense, train_params)

mlflow.set_experiment("mnist-calib")
start_mnist_calib_run(run_name, batch_sizes, model_params_conv, train_params)

# mlflow.set_experiment("mnist-expl")
# model_params_conv["explaining_vars"] = [0, 1, 2]  # rotations, cutoffs, noises
# start_mnist_expl_run(run_name, batch_sizes, model_params_conv, train_params)

# mlflow.set_experiment("dirty-mnist")
# start_dirty_mnist_run(run_name, batch_sizes, model_params_conv, train_params)

# mlflow.set_experiment("cifar10-c-calib")
# model_params_conv["image_shape"] = (3, 32, 32)
# start_cifar10_calib_run(run_name, batch_sizes, model_params_conv, train_params)

# mlflow.set_experiment("cifar10-c_expl")
# corruptions = [
#     "brightness",
#     "contrast",
#     "defocus_blur",
#     "elastic_transform",
#     "fog",  # was broken -> reload?
#     "frost",
#     "gaussian_blur",
#     "gaussian_noise",
#     "glass_blur",
#     "impulse_noise",
#     "jpeg_compression",
#     "motion_blur",
#     "pixelate",
#     "saturate",
#     "shot_noise",
#     "snow",
#     "spatter",
#     "speckle_noise",
#     "zoom_blur",
# ]
# model_params_conv["explaining_vars"] = list(range(len(corruptions)))
# train_params["corruption_levels_train"] = [0, 1, 2]
# start_cifar10_expl_run(run_name, batch_sizes, model_params_conv, train_params)
