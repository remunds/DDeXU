import torch
import numpy as np
from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn as nn

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
    

def train_eval_resnet(train_dl, test_dl, device):
    resnet = ResNetHidden(BasicBlock, [2, 2, 2, 2], num_classes=10)

    resnet.to(device)

    # train NN 
    optimizer = torch.optim.Adam(resnet.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    print("start training resnet")

    for epoch in range(3):
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
    return resnet

def get_latent_dataset(train_ds, train_dl, test_ds, test_dl, resnet, device, batchsize_resnet, save_dir):
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
        np.save(f"{save_dir}/latent_train.npy", latent_train)
        np.save(f"{save_dir}/target_train.npy", target_train)
        print("Latent train dataset shape: ", latent_train.shape)

        for batch_idx, (data, target) in enumerate(test_dl):
            data = data.to(device)
            target = target.to(device)
            low_idx = batch_idx * batchsize_resnet
            high_idx = (batch_idx + 1) * batchsize_resnet
            latent_test[low_idx:high_idx] = resnet.get_hidden(data).squeeze().cpu().numpy()
            target_test[low_idx:high_idx] = target.cpu().numpy()
        np.save(f"{save_dir}/latent_test.npy", latent_test)
        np.save(f"{save_dir}/target_test.npy", target_test)
    print("done collecting latent space dataset")
    return latent_train, target_train, latent_test, target_test


def train_small_mlp(latent_train, target_train, latent_test, target_test, device, batchsize_resnet):
# train simple MLP on latent space for mnist classification to show that latent space is useful

    mlp = nn.Sequential(
        nn.Linear(512, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
        nn.LogSoftmax(dim=1)
    )

    mlp.to(device)

    # train MLP
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    latent_train = torch.from_numpy(latent_train).to(dtype=torch.float32)
    target_train = torch.from_numpy(target_train).to(dtype=torch.long)
    latent_test = torch.from_numpy(latent_test).to(dtype=torch.float32)
    target_test = torch.from_numpy(target_test).to(dtype=torch.long)


    print("start training mlp")
    for epoch in range(2):
        for batch_idx in range(0, latent_train.shape[0], batchsize_resnet):
            optimizer.zero_grad()

            data = latent_train[batch_idx:batch_idx+batchsize_resnet]
            target = target_train[batch_idx:batch_idx+batchsize_resnet]
            data, target = data.to(device), target.to(device)
            output = mlp(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, loss {loss.item()}")

    # evaluate MLP
    mlp.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx in range(0, latent_test.shape[0], batchsize_resnet):
            data = latent_test[batch_idx:batch_idx+batchsize_resnet]
            target = target_test[batch_idx:batch_idx+batchsize_resnet]
            data, target = data.to(device), target.to(device)
            output = mlp(data)
            loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        loss /= latent_test.shape[0] / batchsize_resnet 
        print(f"Test loss: {loss}")
        print(f"Test accuracy: {correct / latent_test.shape[0]}")
