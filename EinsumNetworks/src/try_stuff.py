import torch
import torch.nn as nn
import torch.nn.functional as F
from EinsumNetwork import Graph, EinsumNetwork
# import datasets
from sklearn.datasets import fetch_california_housing

device = 'cuda' if torch.cuda.is_available() else 'cpu'


##########################################################
dataset = 'accidents'

depth = 3
num_repetitions = 10
num_input_distributions = 20
num_sums = 20

max_num_epochs = 10
batch_size = 100
online_em_frequency = 1
online_em_stepsize = 0.05

##########################################################

# # train_x_orig, test_x_orig, valid_x_orig = datasets.load_debd(dataset, dtype='float32')
# data, target = fetch_california_housing(return_X_y=True) # ((20640, 8), (20640,))

# # split 70/20/10
# train_x = data[:14448]
# valid_x = data[14448:18576]
# test_x = data[18576:]
# train_y = target[:14448]
# valid_y = target[14448:18576]
# test_y = target[18576:]

# # to torch
# train_x = torch.from_numpy(train_x).to(torch.device(device), dtype=torch.float32)
# valid_x = torch.from_numpy(valid_x).to(torch.device(device), dtype=torch.float32)
# test_x = torch.from_numpy(test_x).to(torch.device(device), dtype=torch.float32)
# train_y = torch.from_numpy(train_y).to(torch.device(device), dtype=torch.float32)
# valid_y = torch.from_numpy(valid_y).to(torch.device(device), dtype=torch.float32)
# test_y = torch.from_numpy(test_y).to(torch.device(device), dtype=torch.float32)


# train_N, num_dims = train_x.shape
# valid_N = valid_x.shape[0]
# test_N = test_x.shape[0]

# print(train_x.shape) # 12758, 111

from torchvision.datasets import MNIST 
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
     
train_ds = MNIST("mnist", train=True, download=True, transform=ToTensor())
test_ds = MNIST("mnist", train=False, download=True, transform=ToTensor())
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, pin_memory=True, num_workers=1)
test_dl = DataLoader(test_ds, batch_size=64, pin_memory=True, num_workers=1)
print("train_dl", len(train_dl))
print("test_dl", len(test_dl))

print("done loading data")

###############################################################################
# load NN

#TODO: extend ResNet instead of wrapping it -> no need to specify get_hidden

class ResNetHidden(nn.Module):
    """
    ResNet model with the last layer removed
    """
    def __init__(self, original_model):
        super(ResNetHidden, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.original_model = original_model
        # adapt for mnist -> channel=1
        original_model.conv1 = torch.nn.Conv2d(1, 64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3), bias=False)
    
    def get_hidden(self, x):
        for name, module in self.features.named_children():
            if name != "fc":
                x = module(x)
        return x
    
    def forward(self, x):
        self.original_model(x)

from torchvision.models.resnet import resnet18

resnet = ResNetHidden(resnet18(num_classes=10))

# adapt for mnist -> channel=1
# resnet.conv1 = torch.nn.Conv2d(1, 64, 
#     kernel_size=(7, 7), 
#     stride=(2, 2), 
#     padding=(3, 3), bias=False)


resnet.to(device)

# train NN 
optimizer = torch.optim.Adam(resnet.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

print("start training")

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
with torch.no_grad():
    for data, target in test_dl:
        data, target = data.to(device), target.to(device)
        output = resnet(data)
        loss += criterion(output, target).item()
    loss /= len(test_dl)
    print(f"Test loss: {loss}")

##########################################################
# collect latent space dataset
latent = torch.zeros((len(train_dl.dataset), 512), dtype=torch.float32, device=device)
for batch in range(train_x.shape[0] // batch_size):
    low_idx = batch * batch_size
    high_idx = (batch + 1) * batch_size
    latent[low_idx:high_idx] = simple_nn.get_latent(train_x[low_idx:high_idx]) 

print("Latent dataset shape: ", latent.shape)


##########################################################
# EinsumNetwork

# graph = Graph.random_binary_trees(num_var=train_x.shape[1], depth=depth, num_repetitions=num_repetitions)
# graph = Graph.random_binary_trees(num_var=latent.shape[1], depth=depth, num_repetitions=num_repetitions)
graph = Graph.random_binary_trees(num_var=100, depth=depth, num_repetitions=num_repetitions)

args = EinsumNetwork.Args(
    num_classes=1,
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
