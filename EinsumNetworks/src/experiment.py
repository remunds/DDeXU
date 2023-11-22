import torch
import torch.nn as nn
import torch.nn.functional as F
from EinsumNetwork import Graph, EinsumNetwork
import numpy as np
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'


##########################################################

depth = 3
num_repetitions = 20
K = 10

max_num_epochs = 10
batch_size = 100
online_em_frequency = 1
online_em_stepsize = 0.05
exponential_family = EinsumNetwork.NormalArray
exponential_family_args = {'min_var': 1e-6, 'max_var': 0.1}

##########################################################

from torchvision.datasets import MNIST 
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

batchsize_resnet = 64
train_ds = MNIST("mnist", train=True, download=True, transform=ToTensor())
test_ds = MNIST("mnist", train=False, download=True, transform=ToTensor())
train_dl = DataLoader(train_ds, batch_size=batchsize_resnet, shuffle=True, pin_memory=True, num_workers=1)
test_dl = DataLoader(test_ds, batch_size=batchsize_resnet, pin_memory=True, num_workers=1)
print("train_dl", len(train_dl))
print("test_dl", len(test_dl))

print("done loading data")

###############################################################################
exists = os.path.isfile("latent_train.npy") and os.path.isfile("latent_test.npy") and os.path.isfile("target_train.npy") and os.path.isfile("target_test.npy")
exists = False

if exists:
    print("loading latent dataset")
    latent_train = np.load("latent_train.npy")
    target_train = np.load("target_train.npy")
    latent_test = np.load("latent_test.npy")
    target_test = np.load("target_test.npy")
    print("Latent train dataset shape: ", latent_train.shape)

if not exists:
    # load NN
    from ResNetHidden import train_eval_resnet, get_latent_dataset
    resnet = train_eval_resnet(train_dl, test_dl, device) 
    latent_train, target_train, latent_test, target_test = get_latent_dataset(train_ds, train_dl, test_ds, test_dl, resnet, device, batchsize_resnet, save_dir=".")

# train simple MLP on latent space for mnist classification to show that latent space is useful
# from ResNetHidden import train_small_mlp
# train_small_mlp(latent_train, target_train, latent_test, target_test, device, batchsize_resnet)

# normalize latent space
latent_train /= latent_train.max()
latent_test /= latent_test.max()
latent_train -= .5
latent_test -= .5

latent_train = torch.from_numpy(latent_train).to(dtype=torch.float32)
target_train = torch.from_numpy(target_train).to(dtype=torch.long)
latent_test = torch.from_numpy(latent_test).to(dtype=torch.float32)
target_test = torch.from_numpy(target_test).to(dtype=torch.long)


##########################################################
# EinsumNetwork

graph = Graph.random_binary_trees(num_var=latent_train.shape[1], depth=depth, num_repetitions=num_repetitions)

args = EinsumNetwork.Args(
        num_var=latent_train.shape[1],
        num_dims=1,
        num_classes=10,
        num_sums=K,
        num_input_distributions=K,
        exponential_family=exponential_family,
        exponential_family_args=exponential_family_args,
        online_em_frequency=online_em_frequency,
        online_em_stepsize=online_em_stepsize
        )

einet = EinsumNetwork.EinsumNetwork(graph, args)
einet.initialize()
einet.to(device)
print(einet)

latent_train_N = latent_train.shape[0]
latent_test_N = latent_test.shape[0]

latent_train = latent_train.to(device)
latent_test = latent_test.to(device)
target_train = target_train.to(device)
target_test = target_test.to(device)


# TODO: perform different perturbations on MNIST
# rotation, translation, scaling, noise, occlusion, etc.
# store perturbation in corresponding variables
# add perturbations as variables to EiNet
# expect to see lower LL for perturbed data
# expect to see low LL for variable that corresponds to the actual perturbation 
# -> Explanation

ood = torch.rand(latent_test.shape[0], latent_test.shape[1]).to(device)

for epoch_count in range(max_num_epochs):
    if epoch_count % 2 == 0:
        # evaluate
        train_ll = EinsumNetwork.eval_loglikelihood_batched(einet, latent_train)
        test_ll = EinsumNetwork.eval_loglikelihood_batched(einet, latent_test)
        ood_ll = EinsumNetwork.eval_loglikelihood_batched(einet, ood)

        print("[{}]   train LL {}  test LL {} ood LL {}".format(epoch_count,
                                                                    train_ll / latent_train_N,
                                                                    test_ll / latent_test_N,
                                                                    ood_ll / latent_train_N))
        
        print("train accuracy: ", EinsumNetwork.eval_accuracy_batched(einet, latent_train, target_train, batch_size))

    # train
    idx_batches = torch.randperm(latent_train_N).split(batch_size)
    for batch_count, idx in enumerate(idx_batches):
        batch_x = latent_train[idx, :]
        outputs = einet.forward(batch_x)
        ll_sample = EinsumNetwork.log_likelihoods(outputs, target_train[idx])
        log_likelihood = ll_sample.sum()
        objective = log_likelihood
        objective.backward()
        einet.em_process_batch()
    einet.em_update()
