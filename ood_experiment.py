from shift_dev import SHIFTDataset
from shift_dev.types import Keys
from shift_dev.utils.backend import ZipBackend
from torch.utils.data import DataLoader
from torchvision import transforms

from simple_einet.einet import Einet, EinetConfig
from simple_einet.layers.distributions.normal import Normal, RatNormal
import torch

preprocess = transforms.Compose(
    [
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def collate(batch):
    # a batch consists of a list of dicts
    # each dict has 'front', then 'images', 'boxes2d' and 'boxes2d_classes'
    # so each dict is a single image

    # model requires a list of images and targets
    # targets is a list of dicts with 'boxes' and 'labels' (one dict per image)
    images = []
    targets = []
    for b in batch:
        images.append(preprocess(b["front"]["images"]))
        targets.append(
            {"boxes": b["front"]["boxes2d"], "labels": b["front"]["boxes2d_classes"]}
        )
    images = torch.stack(images)
    return images, targets


def get_shift_loader():
    dataset = SHIFTDataset(
        data_root="/data_docker/datasets/shift/",
        split="train",
        keys_to_load=[
            Keys.images,
            # Keys.intrinsics,
            Keys.boxes2d,
            # Keys.boxes2d_classes,
            # Keys.boxes2d_track_ids,
            # Keys.segmentation_masks,
        ],
        views_to_load=["front"],
        framerate="images",
        shift_type="discrete",
        backend=ZipBackend(),
        verbose=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=6,  # 6 for resnet50,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
    )
    return dataloader


from torchvision.models.resnet import resnet18


from torchvision.models.detection.backbone_utils import (
    _resnet_fpn_extractor,
    _validate_trainable_layers,
)

from torchvision.models.detection.faster_rcnn import (
    FasterRCNN,
    FastRCNNPredictor,
)


class FastRCNNPredictorEinet(FastRCNNPredictor):
    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictorEinet, self).__init__(in_channels, num_classes)
        self.pre_classifier = torch.nn.Linear(in_channels, 128)
        self.cls_score = torch.nn.Linear(128, num_classes)
        self.bbox_pred = torch.nn.Linear(128, num_classes * 4)
        self.divisor = 128  # in_channels
        self.einet = Einet(
            EinetConfig(
                num_features=128,
                num_channels=1,
                depth=5,
                num_sums=11,
                num_leaves=15,
                num_repetitions=10,
                num_classes=1,
                leaf_type=RatNormal,
                leaf_kwargs={
                    "min_sigma": 0.00001,
                    "max_sigma": 10.0,
                },
                layer_type="einsum",
                dropout=0.0,
            )
        )
        self.einet_active = False

    def forward(self, x):
        # x-shape: (512, 1024)
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
            )
        x = x.flatten(start_dim=1)

        x = self.pre_classifier(x)
        if self.einet_active:
            self.einet_loss = -self.einet(x).mean() / self.divisor  # NLL loss

        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


def fasterrcnn_resnet18_fpn(**kwargs) -> FasterRCNN:
    """
    Constructs a FasterRCNN model with a ResNet-50-FPN backbone.
    """
    num_classes = 6  # car, truck, bus, bicycle, motorcycle, and pedestrian
    norm_layer = torch.nn.BatchNorm2d

    backbone = resnet18(norm_layer=norm_layer)
    trainable_backbone_layers = _validate_trainable_layers(False, None, 5, 3)
    backbone = _resnet_fpn_extractor(
        backbone,
        trainable_backbone_layers,
    )
    representation_size = 1024
    box_predictor = FastRCNNPredictorEinet(representation_size, num_classes)
    # model = FasterRCNN(backbone, num_classes, box_predictor=box_predictor, **kwargs)
    model = FasterRCNN(backbone, box_predictor=box_predictor, **kwargs)
    return model


model = fasterrcnn_resnet18_fpn()
print(model)

device = "cuda"
model.to(device)

from tqdm import tqdm

# TODO: How well does the model work with and without the pre-classifier?
# Does it negatively impact the training?
# I could also leave it empty, since the training is in two stages, it shouldnt be too big of a deal
# Maybe also try training with combined loss (for classification)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 10
dl = get_shift_loader()
for epoch in range(epochs):
    losses = []
    for data, target in tqdm(dl):
        optimizer.zero_grad()
        data = data.to(device)
        target = [{k: v.to(device) for k, v in t.items()} for t in target]
        # remove useless first dimension
        data = data.squeeze(1)
        output = model(data, target)
        loss = sum(loss for loss in output.values())
        if model.roi_heads.box_predictor.einet_active:
            einet_loss = model.roi_heads.box_predictor.einet_loss
            loss += einet_loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"Epoch {epoch}: {sum(losses) / len(losses)}")
