import torch
import torch.nn as nn
import torch.nn.functional as F
from EinsumNetwork import Graph, EinsumNetwork
import numpy as np
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'


##########################################################

depth = 3
num_repetitions = 10
num_input_distributions = 20
num_sums = 20

max_num_epochs = 10
batch_size = 100
online_em_frequency = 1
online_em_stepsize = 0.05

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

if exists:
    print("loading latent dataset")
    latent_train = np.load("latent_train.npy")
    target_train = np.load("target_train.npy")
    latent_test = np.load("latent_test.npy")
    target_test = np.load("target_test.npy")
    print("Latent train dataset shape: ", latent_train.shape)

if not exists:
    # load NN

    #TODO: extend ResNet instead of wrapping it -> no need to specify get_hidden
    from torchvision.models.resnet import ResNet, BasicBlock

    class ResNetHidden(ResNet):
        """
        ResNet model with ability to return latent space 
        """
        def __init__(self, *args, **kwargs):
            super(ResNetHidden, self).__init__(*args, **kwargs)
            # adapt for mnist -> channel=1
            self.conv1 = torch.nn.Conv2d(1, 64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3), bias=False)
        
        def get_hidden(self, x):
            for name, module in self.named_children():
                if name != "fc":
                    x = module(x)
            return x

    resnet = ResNetHidden(BasicBlock, [2, 2, 2, 2], num_classes=10)

    resnet.to(device)

    # train NN 
    optimizer = torch.optim.Adam(resnet.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    print("start training resnet")

    for epoch in range(1):
        for data, target in train_dl:
            optimizer.zero_grad()

            data, target = data.to(device), target.to(device)
            output = resnet(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step() 
        print(f"Epoch {epoch}, loss {loss.item()}")

    # evaluate NN
    resnet.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_dl:
            data, target = data.to(device), target.to(device)
            output = resnet(data)
            loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        loss /= len(test_dl)
        print(f"Test loss: {loss}")
        print(f"Test accuracy: {correct / len(test_dl.dataset)}")

    ##########################################################
    # collect latent space dataset
    latent_train = np.zeros((train_ds.data.shape[0], 512))
    target_train = np.zeros((train_ds.data.shape[0],))
    latent_test = np.zeros((test_ds.data.shape[0], 512))
    target_test = np.zeros((test_ds.data.shape[0],))
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_dl):
            data = data.to(device)
            target = target.to(device)
            low_idx = batch_idx * batchsize_resnet
            high_idx = (batch_idx + 1) * batchsize_resnet
            latent_train[low_idx:high_idx] = resnet.get_hidden(data).squeeze().cpu().numpy()
            target_train[low_idx:high_idx] = target.cpu().numpy()
        np.save("latent_train.npy", latent_train)
        np.save("target_train.npy", target_train)
        print("Latent train dataset shape: ", latent_train.shape)

        for batch_idx, (data, target) in enumerate(test_dl):
            data = data.to(device)
            target = target.to(device)
            low_idx = batch_idx * batchsize_resnet
            high_idx = (batch_idx + 1) * batchsize_resnet
            latent_test[low_idx:high_idx] = resnet.get_hidden(data).squeeze().cpu().numpy()
            target_test[low_idx:high_idx] = target.cpu().numpy()
        np.save("latent_test.npy", latent_test)
        np.save("target_test.npy", target_test)
    print("done collecting latent space dataset")


##########################################################
# train simple MLP on latent space for mnist classification to show that latent space is useful
# mlp = nn.Sequential(
#     nn.Linear(512, 100),
#     nn.ReLU(),
#     nn.Linear(100, 10),
#     nn.LogSoftmax(dim=1)
# )

# mlp.to(device)

# # train MLP
# optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
# criterion = torch.nn.CrossEntropyLoss()

# latent_train = torch.from_numpy(latent_train).to(dtype=torch.float32)
# target_train = torch.from_numpy(target_train).to(dtype=torch.long)
# latent_test = torch.from_numpy(latent_test).to(dtype=torch.float32)
# target_test = torch.from_numpy(target_test).to(dtype=torch.long)


# print("start training mlp")
# for epoch in range(10):
#     for batch_idx in range(0, latent_train.shape[0], batchsize_resnet):
#         optimizer.zero_grad()

#         data = latent_train[batch_idx:batch_idx+batchsize_resnet]
#         target = target_train[batch_idx:batch_idx+batchsize_resnet]
#         data, target = data.to(device), target.to(device)
#         output = mlp(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#     print(f"Epoch {epoch}, loss {loss.item()}")

# # evaluate MLP
# mlp.eval()
# loss = 0
# correct = 0
# with torch.no_grad():
#     for batch_idx in range(0, latent_test.shape[0], batchsize_resnet):
#         data = latent_test[batch_idx:batch_idx+batchsize_resnet]
#         target = target_test[batch_idx:batch_idx+batchsize_resnet]
#         data, target = data.to(device), target.to(device)
#         output = mlp(data)
#         loss += criterion(output, target).item()
#         pred = output.argmax(dim=1, keepdim=True)
#         correct += pred.eq(target.view_as(pred)).sum().item()

#     loss /= len(test_dl)
#     print(f"Test loss: {loss}")
#     print(f"Test accuracy: {correct / len(test_dl.dataset)}")

##########################################################
# EinsumNetwork

# graph = Graph.random_binary_trees(num_var=train_x.shape[1], depth=depth, num_repetitions=num_repetitions)
# graph = Graph.random_binary_trees(num_var=latent.shape[1], depth=depth, num_repetitions=num_repetitions)
graph = Graph.random_binary_trees(num_var=100, depth=depth, num_repetitions=num_repetitions)

args = EinsumNetwork.Args(
    num_classes=10,
    num_input_distributions=num_input_distributions,
    exponential_family=EinsumNetwork.CategoricalArray,
    exponential_family_args={'K': 2},
    num_sums=num_sums,
    # num_var=train_x.shape[1],
    num_var=100,
    online_em_frequency=1,
    online_em_stepsize=0.05)

einet = EinsumNetwork.EinsumNetwork(graph, args)
einet.initialize()
einet.to(device)
print(einet)

for epoch_count in range(max_num_epochs):

    # evaluate
    train_ll = EinsumNetwork.eval_loglikelihood_batched(einet, train_x)
    valid_ll = EinsumNetwork.eval_loglikelihood_batched(einet, valid_x)
    test_ll = EinsumNetwork.eval_loglikelihood_batched(einet, test_x)

    print("[{}]   train LL {}   valid LL {}  test LL {}".format(epoch_count,
                                                                train_ll / train_N,
                                                                valid_ll / valid_N,
                                                                test_ll / test_N))

    # train
    idx_batches = torch.randperm(train_N).split(batch_size)
    for batch_count, idx in enumerate(idx_batches):
        batch_x = train_x[idx, :]
        outputs = einet.forward(batch_x)

        ll_sample = EinsumNetwork.log_likelihoods(outputs)
        log_likelihood = ll_sample.sum()

        objective = log_likelihood
        objective.backward()

        einet.em_process_batch()

    einet.em_update()
