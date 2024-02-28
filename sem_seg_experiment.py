# just do scene level uncertainty?
# but: only generative training -> should still be good enough for LL's
# and LL explanations
# no more aleatoric uncert though...

# or does H*W*C as output-classes make sense? -> Then gen+discr training possible
# -> Is the underlying SPN shared for all outputs? -> seems like it is, only last layer
# then training via reconstruction or iou loss possible?
# -> TRY
# then entropy can be computed pixelwise (each class)

from shift_dev import SHIFTDataset
from shift_dev.types import Keys
from shift_dev.utils.backend import ZipBackend
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_v2_s

from simple_einet.einet import Einet, EinetConfig
from simple_einet.layers.distributions.normal import Normal
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
        batch_size=1,  # 6 for resnet50,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
    )
    return dataloader


from torchvision.models.detection.backbone_utils import BackboneWithFPN


from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.models.detection import RetinaNet
from torchvision.models.resnet import resnet18


from torchvision.models.detection.backbone_utils import (
    _resnet_fpn_extractor,
    _validate_trainable_layers,
)


from torchvision.ops.feature_pyramid_network import LastLevelP6P7


def retinanet_resnet18_fpn(**kwargs) -> RetinaNet:
    """
    Constructs a RetinaNet model with a ResNet-50-FPN backbone.
    """
    num_classes = 3
    norm_layer = torch.nn.BatchNorm2d

    # backbone = resnet50(
    #     weights=weights_backbone, progress=progress, norm_layer=norm_layer
    # )
    backbone = resnet18(norm_layer=norm_layer)
    trainable_backbone_layers = _validate_trainable_layers(False, None, 5, 3)
    # skip P2 because it generates too many anchors (according to their paper)
    backbone = _resnet_fpn_extractor(
        backbone,
        trainable_backbone_layers,
        returned_layers=[2, 3, 4],
        extra_blocks=LastLevelP6P7(256, 256),
    )
    model = RetinaNet(backbone, num_classes, **kwargs)

    return model


def efficientnet_fpn(backbone):
    return_layers = {"features": 0}
    in_channels_list = [1280]
    out_channels = 256
    extra_blocks = LastLevelMaxPool()

    return BackboneWithFPN(
        backbone,
        return_layers,
        in_channels_list,
        out_channels,
        extra_blocks=extra_blocks,
        norm_layer=None,
    )


def retinanet_efficientnet_fpn(num_classes, **kwargs):
    """
    Constructs a RetinaNet model with a EfficientNet_v2_s-FPN backbone.
    """

    # backbone = resnet50(weights=weights_backbone, progress=progress, norm_layer=norm_layer)
    backbone = efficientnet_v2_s()
    # skip P2 because it generates too many anchors (according to their paper)
    backbone = efficientnet_fpn(
        backbone,
    )
    model = RetinaNet(backbone, num_classes, **kwargs)

    return model


from torchvision.models.detection import retinanet_resnet50_fpn

# model = retinanet_resnet50_fpn()
model = retinanet_resnet18_fpn()
# model: RetinaNet = retinanet_efficientnet_fpn(num_classes=10)
print(model)
# TODO: this still triggers device side assertion errors in CUDA

# in_features = in_channels * width * height

# cfg = EinetConfig(
#     num_features=in_features,
#     num_channels=1,
#     depth=1,
#     num_sums=5,
#     num_leaves=5,
#     num_repetitions=5,
#     num_classes=out_channels * height * width,
#     # leaf_type=self.einet_leaf_type,
#     leaf_type=Normal,
#     # leaf_kwargs=leaf_kwargs,
#     layer_type="einsum",
#     dropout=0.0,
# )
# einet = Einet(cfg)
# print(einet)
device = "cuda"
model.to(device)
# model.eval()
from tqdm import tqdm

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 10
dl = get_shift_loader()
for epoch in range(epochs):
    losses = []
    for data, target in tqdm(dl):
        optimizer.zero_grad()
        data = data.to(device)
        target = [{k: v.to(device) for k, v in t.items()} for t in target]
        # remove first dimension
        data = data.squeeze(1)
        # print(target.shape)
        output = model(data, target)
        cls_loss = output["classification"]
        reg_loss = output["bbox_regression"]
        loss = cls_loss + reg_loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"Epoch {epoch}: {sum(losses) / len(losses)}")
