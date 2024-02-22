# quick OD test using Pascal VOC

from torchvision.datasets import VOCDetection
from torchvision import transforms

test_transformer = transforms.Compose(
    [
        transforms.ToTensor(),
        # this is random -> need to fix
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

# load data
voc_train_ds = VOCDetection(
    "/data_docker/datasets/voc",
    image_set="train",
    download=True,
    transform=test_transformer,
)
voc_valid_ds = VOCDetection(
    "/data_docker/datasets/voc",
    image_set="val",
    download=True,
    transform=test_transformer,
)

print(len(voc_train_ds))

# load retinanet
from torchvision.models.detection import retinanet_resnet50_fpn

retinanet = retinanet_resnet50_fpn(pretrained=True)
retinanet.eval()

# load data
from torch.utils.data import DataLoader

voc_train_dl = DataLoader(
    voc_train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
)
voc_valid_dl = DataLoader(
    voc_valid_ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=True
)

# train retinanet
import torch
import torch.optim as optim
import torchvision.transforms as T
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from tqdm import tqdm

# load pretrained model
backbone = resnet_fpn_backbone("resnet50", pretrained=True)
model = RetinaNet(
    backbone,
    num_classes=20,
    min_size=800,
    max_size=1333,
    image_mean=[0.485, 0.456, 0.406],
    image_std=[0.229, 0.224, 0.225],
    rpn_anchor_generator=AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    ),
)

# move to GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# train
for epoch in range(10):
    model.train()
    for images, targets in tqdm(voc_train_dl):
        # images = list(image.to(device) for image in images)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        print(type(images))
        print(type(targets))
        print(targets.shape)
        print(images.shape)
        print(targets.shape)
        images = images.to(device)
        targets = targets.to(device)

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    # update learning rate
    lr_scheduler.step()

    # evaluate on the test dataset
    model.eval()
    for images, targets in tqdm(voc_valid_dl):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # no gradients
        with torch.no_grad():
            prediction = model(images)

    print(f"Epoch {epoch} finished")
