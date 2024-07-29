import torch
from torch import nn
import torch.nn.functional as F

import mlflow

from torchvision.models import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l
from torch.nn.utils.parametrizations import spectral_norm as spectral_norm_torch

# from spectral_normalization import spectral_norm

from simple_einet.layers.distributions.normal import CustomNormal, Normal, RatNormal
from simple_einet.layers.distributions.categorical import Categorical
from simple_einet.layers.distributions.multidistribution import MultiDistributionLayer
from simple_einet.einet import EinetConfig, Einet

from utils.einet_utils import EinetUtils
from utils.gmm_utils import GMMUtils
from utils.sngp_utils import SNGPUtils


class EfficientNetDet(nn.Module, EinetUtils):
    """
    EfficientNetV2-S. Optionally spectral normalized.
    """

    def __init__(
        self,
        num_classes,
        image_shape,  # (C, H, W)
        explaining_vars=[],  # indices of variables that should be explained
        spec_norm_bound=0.9,
        num_hidden=32,
        spectral_normalization=False,
        model_size="s",
        **kwargs,
    ):
        super(EfficientNetDet, self).__init__()
        self.num_classes = num_classes
        self.explaining_vars = explaining_vars
        self.spectral_normalization = spectral_normalization
        self.spec_norm_bound = spec_norm_bound
        self.image_shape = image_shape
        self.marginalized_scopes = None
        # self.num_hidden = 1280  # from efficientnet_s
        self.num_hidden = num_hidden
        self.model_size = model_size
        self.backbone = self.make_efficientnet()

    def make_efficientnet(self):
        def replace_layers_rec(layer):
            """Recursively apply spectral normalization to Conv and Linear layers."""
            if len(list(layer.children())) == 0:
                if isinstance(layer, torch.nn.Conv2d):
                    layer = spectral_norm_torch(layer)
                    # layer = spectral_norm(layer, norm_bound=self.spec_norm_bound)
                elif isinstance(layer, torch.nn.Linear):
                    layer = spectral_norm_torch(layer)
                    # layer = spectral_norm(layer, norm_bound=self.spec_norm_bound)
            else:
                for child in list(layer.children()):
                    replace_layers_rec(child)

        if self.model_size == "s":
            model = efficientnet_v2_s()
            out_channels = 24
        elif self.model_size == "m":
            model = efficientnet_v2_m()
        elif self.model_size == "l":
            model = efficientnet_v2_l()
            out_channels = 32
        else:
            raise NotImplementedError

        model.features[0][0] = torch.nn.Conv2d(
            self.image_shape[0],
            out_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
        )
        # model.classifier = torch.nn.Linear(1280, self.num_classes)
        model.pre_classifier = torch.nn.Linear(1280, self.num_hidden)
        model.classifier = torch.nn.Linear(self.num_hidden, self.num_classes)
        # apply spectral normalization
        if self.spectral_normalization:
            replace_layers_rec(model)
        return model

    def forward_hidden(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.pre_classifier(x)
        return x

    def forward(self, x):
        # x is flattened
        # extract explaining vars
        exp_vars = x[:, self.explaining_vars]
        # mask out explaining vars for backbone
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:, self.explaining_vars] = False
        x = x[mask]
        # reshape to image
        x = x.reshape(-1, self.image_shape[0], self.image_shape[1], self.image_shape[2])

        # feed through backbone
        x = self.forward_hidden(x)
        self.hidden = x

        return self.backbone.classifier(x)  # default classifier

    def activate_uncert_head(self, deactivate_backbone=True):
        pass

    def deactivate_uncert_head(self):
        pass


class EfficientNetSPN(nn.Module, EinetUtils):
    """
    Spectral normalized EfficientNetV2-S with einet as the output layer.
    """

    def __init__(
        self,
        num_classes,
        image_shape,  # (C, H, W)
        explaining_vars=[],  # indices of variables that should be explained
        spec_norm_bound=0.9,
        einet_depth=3,
        einet_num_sums=20,
        einet_num_leaves=20,
        einet_num_repetitions=1,
        einet_leaf_type="Normal",
        einet_dropout=0.0,
        num_hidden=32,
        model_size="s",
        **kwargs,
    ):
        super(EfficientNetSPN, self).__init__()
        self.num_classes = num_classes
        self.explaining_vars = explaining_vars
        self.spec_norm_bound = spec_norm_bound
        self.einet_depth = einet_depth
        self.einet_num_sums = einet_num_sums
        self.einet_num_leaves = einet_num_leaves
        self.einet_num_repetitions = einet_num_repetitions
        if einet_leaf_type == "Normal":
            # self.einet_leaf_type = Normal
            self.einet_leaf_type = RatNormal
        else:
            raise NotImplementedError
        self.einet_dropout = einet_dropout
        self.image_shape = image_shape
        self.marginalized_scopes = None
        # self.num_hidden = 1280  # from efficientnet_s
        self.num_hidden = num_hidden
        self.model_size = model_size
        self.backbone = self.make_efficientnet()
        self.einet = self.make_einet_output_layer(
            self.num_hidden + len(explaining_vars), num_classes
        )
        self.mean = None
        self.std = None

    def make_efficientnet(self):
        def replace_layers_rec(layer):
            """Recursively apply spectral normalization to Conv and Linear layers."""
            if len(list(layer.children())) == 0:
                if isinstance(layer, torch.nn.Conv2d):
                    layer = spectral_norm_torch(layer)
                    # layer = spectral_norm(layer, norm_bound=self.spec_norm_bound)
                elif isinstance(layer, torch.nn.Linear):
                    layer = spectral_norm_torch(layer)
                    # layer = spectral_norm(layer, norm_bound=self.spec_norm_bound)
            else:
                for child in list(layer.children()):
                    replace_layers_rec(child)

        if self.model_size == "s":
            model = efficientnet_v2_s()
            out_channels = 24
        elif self.model_size == "m":
            model = efficientnet_v2_m()
        elif self.model_size == "l":
            out_channels = 32
            model = efficientnet_v2_l()
        else:
            raise NotImplementedError

        model.features[0][0] = torch.nn.Conv2d(
            self.image_shape[0],
            out_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
        )
        # model.classifier = torch.nn.Linear(1280, self.num_classes)
        model.pre_classifier = torch.nn.Linear(1280, self.num_hidden)
        model.classifier = torch.nn.Linear(self.num_hidden, self.num_classes)
        # apply spectral normalization
        replace_layers_rec(model)
        return model

    def activate_uncert_head(self, deactivate_backbone=True):
        """
        Activates the einet output layer for second stage training and inference.
        """
        self.einet_active = True
        if deactivate_backbone:
            for param in self.parameters():
                param.requires_grad = False
        else:
            for param in self.parameters():
                param.requires_grad = True
        for param in self.einet.parameters():
            param.requires_grad = True

    def deactivate_uncert_head(self):
        """
        Deactivates the einet output layer for first stage training.
        """
        self.einet_active = False
        for param in self.einet.parameters():
            param.requires_grad = False
            pass

    def make_einet_output_layer(self, in_features, out_features):
        """Uses einet as the output layer."""
        if len(self.explaining_vars) > 0:
            leaf_type = MultiDistributionLayer
            print("expl vars: ", self.explaining_vars)
            print("in_features: ", in_features)
            print("out_features: ", out_features)
            scopes_a = torch.arange(0, len(self.explaining_vars))
            scopes_b = torch.arange(len(self.explaining_vars), in_features)
            leaf_kwargs = {
                "scopes_to_dist": [
                    (
                        scopes_a,
                        # Categorical,
                        # {"num_bins": 4},
                        RatNormal,
                        {
                            "min_sigma": 0.00001,
                            "max_sigma": 10.0,
                        },
                    ),
                    # (scopes_a, Categorical, {"num_bins": 6}),
                    (
                        scopes_b,
                        RatNormal,
                        {"min_sigma": 0.00001, "max_sigma": 10.0},
                    ),  # Tuple of (scopes, class, kwargs)
                ]
            }
        else:
            leaf_type = RatNormal
            leaf_kwargs = {"min_sigma": 0.00001, "max_sigma": 10.0}
        # leaf_type = RatNormal
        # leaf_kwargs = {"min_sigma": 0.00001, "max_sigma": 50.0}

        mlflow.log_param("leaf_type", leaf_type)
        mlflow.log_param("leaf_kwargs", leaf_kwargs)

        cfg = EinetConfig(
            num_features=in_features,
            num_channels=1,
            depth=self.einet_depth,
            num_sums=self.einet_num_sums,
            num_leaves=self.einet_num_leaves,
            num_repetitions=self.einet_num_repetitions,
            num_classes=out_features,
            # leaf_type=self.einet_leaf_type,
            leaf_type=leaf_type,
            leaf_kwargs=leaf_kwargs,
            layer_type="einsum",
            dropout=self.einet_dropout,
        )
        model = Einet(cfg)
        return model

    def forward_hidden(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.pre_classifier(x)
        return x

    def forward(self, x):
        # x is flattened
        # extract explaining vars
        exp_vars = x[:, self.explaining_vars]
        # mask out explaining vars for resnet
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:, self.explaining_vars] = False
        x = x[mask]
        # reshape to image
        x = x.reshape(-1, self.image_shape[0], self.image_shape[1], self.image_shape[2])

        # feed through resnet
        x = self.forward_hidden(x)
        self.hidden = x

        if self.einet_active:
            # discrete -> continuous
            # exp_vars = self.dequantize(exp_vars)
            # classifier is einet, so we need to concatenate the explaining vars
            x = torch.cat([exp_vars, x], dim=1)
            # normalize the input to the einet
            # this requires self.mean and self.std to be set
            x = self.normalize(x)
            self.einet_input = x
            return self.einet(x, marginalized_scopes=self.marginalized_scopes)

        return self.backbone.classifier(x)  # default classifier


class EfficientNetGMM(nn.Module, GMMUtils, EinetUtils):
    """
    Spectral normalized EfficientNetV2-S with gmm as the output layer.
    See DDU paper for details.
    """

    def __init__(
        self,
        num_classes,
        image_shape,  # (C, H, W)
        explaining_vars=[],  # indices of variables that should be explained
        spec_norm_bound=0.9,
        num_hidden=32,
        **kwargs,
    ):
        super(EfficientNetGMM, self).__init__()
        self.num_classes = num_classes
        self.explaining_vars = explaining_vars
        self.spec_norm_bound = spec_norm_bound
        self.image_shape = image_shape
        self.marginalized_scopes = None
        self.num_hidden = num_hidden
        self.backbone = self.make_efficientnet()
        self.gmm = None
        self.gmm_active = False

    def make_efficientnet(self):
        def replace_layers_rec(layer):
            """Recursively apply spectral normalization to Conv and Linear layers."""
            if len(list(layer.children())) == 0:
                if isinstance(layer, torch.nn.Conv2d):
                    layer = spectral_norm_torch(layer)
                    # layer = spectral_norm(layer, norm_bound=self.spec_norm_bound)
                elif isinstance(layer, torch.nn.Linear):
                    layer = spectral_norm_torch(layer)
                    # layer = spectral_norm(layer, norm_bound=self.spec_norm_bound)
            else:
                for child in list(layer.children()):
                    replace_layers_rec(child)

        model = efficientnet_v2_s()
        model.features[0][0] = torch.nn.Conv2d(
            self.image_shape[0],
            24,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
        )
        model.pre_classifier = torch.nn.Linear(1280, self.num_hidden)
        model.classifier = torch.nn.Linear(self.num_hidden, self.num_classes)
        # apply spectral normalization
        replace_layers_rec(model)
        return model

    def forward_hidden(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.pre_classifier(x)
        return x

    def forward(self, x):
        # x is flattened
        # extract explaining vars
        exp_vars = x[:, self.explaining_vars]
        # mask out explaining vars for resnet
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:, self.explaining_vars] = False
        x = x[mask]
        # reshape to image
        x = x.reshape(-1, self.image_shape[0], self.image_shape[1], self.image_shape[2])

        # feed through resnet
        x = self.forward_hidden(x)
        self.hidden = x

        if self.gmm_active and self.gmm is not None:
            # notebook: logsumexp(logits, dim=1)
            # logits = gmm_evaluate = log_prob of all datapoints of the embedding
            return self.gmm_logits(x)

        return self.backbone.classifier(x)  # default classifier


from utils.sngp import Laplace
from utils.sngp_utils import SNGPUtils


class EfficientNetSNGP(nn.Module, SNGPUtils, EinetUtils):
    """
    Spectral normalized EfficientNetV2-S with einet as the output layer.
    """

    def __init__(
        self,
        num_classes,
        image_shape,  # (C, H, W)
        train_batch_size,
        train_num_data,
        explaining_vars=[],  # indices of variables that should be explained
        spec_norm_bound=0.9,
        num_hidden=32,  # 32 worked well, already fails with 128
        **kwargs,
    ):
        super(EfficientNetSNGP, self).__init__()
        self.num_classes = num_classes
        self.explaining_vars = explaining_vars
        self.spec_norm_bound = spec_norm_bound
        self.image_shape = image_shape
        self.marginalized_scopes = None
        self.num_hidden = num_hidden
        self.backbone = self.make_efficientnet()
        self.train_batch_size = train_batch_size
        self.train_num_data = train_num_data
        self.sngp = self.make_sngp_output_layer(self.num_hidden + len(explaining_vars))

    def make_efficientnet(self):
        def replace_layers_rec(layer):
            """Recursively apply spectral normalization to Conv and Linear layers."""
            if len(list(layer.children())) == 0:
                if isinstance(layer, torch.nn.Conv2d):
                    layer = spectral_norm_torch(layer)
                    # layer = spectral_norm(layer, norm_bound=self.spec_norm_bound)
                elif isinstance(layer, torch.nn.Linear):
                    layer = spectral_norm_torch(layer)
                    # layer = spectral_norm(layer, norm_bound=self.spec_norm_bound)
            else:
                for child in list(layer.children()):
                    replace_layers_rec(child)

        model = efficientnet_v2_s()
        model.features[0][0] = torch.nn.Conv2d(
            self.image_shape[0],
            24,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
        )
        model.pre_classifier = torch.nn.Linear(1280, self.num_hidden)
        model.classifier = torch.nn.Linear(self.num_hidden, self.num_classes)
        # apply spectral normalization
        replace_layers_rec(model)
        return model

    def activate_uncert_head(self, deactivate_backbone=True):
        """ """
        return

    def deactivate_uncert_head(self):
        """ """
        return

    def make_sngp_output_layer(self, in_features):
        """Uses sngp as the output layer."""

        # dummy feature extractor, not used
        def dummy_feature_extractor(x):
            return x

        sngp_hps = dict(
            num_deep_features=in_features,
            num_gp_features=128,
            normalize_gp_features=False,
            num_random_features=512,
            num_outputs=self.num_classes,
            num_data=self.train_num_data,
            train_batch_size=self.train_batch_size,
            ridge_penalty=1,
            feature_scale=2,
            mean_field_factor=25,
        )
        mlflow.log_params(sngp_hps)

        model = Laplace(
            dummy_feature_extractor,
            **sngp_hps,
        )
        return model

    def forward_hidden(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.pre_classifier(x)
        return x

    def forward(self, x):
        # x is flattened
        # extract explaining vars
        exp_vars = x[:, self.explaining_vars]
        # mask out explaining vars for resnet
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:, self.explaining_vars] = False
        x = x[mask]
        # reshape to image
        x = x.reshape(-1, self.image_shape[0], self.image_shape[1], self.image_shape[2])

        # feed through resnet
        x = self.forward_hidden(x)
        self.hidden = x

        return self.sngp(x)


from torchvision.ops.stochastic_depth import stochastic_depth


class StochasticDepthActive(nn.Module):
    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input):
        # force training to be True
        return stochastic_depth(input, self.p, self.mode, training=True)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(p={self.p}, mode={self.mode})"
        return s


class EfficientNetDropout(nn.Module, EinetUtils):
    """
    Monte Carlo Droupoiut EfficientNet.
    For training, keep uncert_head deactivated, activate for sampling-inference
    """

    def __init__(
        self,
        num_classes,
        image_shape,  # (C, H, W)
        explaining_vars=[],  # indices of variables that should be explained
        spec_norm_bound=0.9,
        num_hidden=1280,
        spectral_normalization=False,
        **kwargs,
    ):
        super(EfficientNetDropout, self).__init__()
        self.num_classes = num_classes
        self.explaining_vars = explaining_vars
        self.spec_norm_bound = spec_norm_bound
        self.image_shape = image_shape
        self.marginalized_scopes = None
        self.num_hidden = num_hidden
        self.spectral_normalization = spectral_normalization
        self.backbone = self.make_efficientnet()

    def make_efficientnet(self):
        def replace_layers_rec(layer):
            import torchvision

            """Recursively apply spectral normalization to Conv and Linear layers."""
            if len(list(layer.children())) == 0:
                if isinstance(layer, torch.nn.Conv2d):
                    layer = spectral_norm_torch(layer)
                    # layer = spectral_norm(layer, norm_bound=self.spec_norm_bound)
                elif isinstance(layer, torch.nn.Linear):
                    layer = spectral_norm_torch(layer)
                    # layer = spectral_norm(layer, norm_bound=self.spec_norm_bound)
                elif isinstance(layer, torchvision.ops.StochasticDepth):
                    layer = StochasticDepthActive(layer.p, layer.mode)
            else:
                for child in list(layer.children()):
                    replace_layers_rec(child)

        model = efficientnet_v2_s()
        model.features[0][0] = torch.nn.Conv2d(
            self.image_shape[0],
            24,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
        )
        model.pre_classifier = torch.nn.Linear(1280, self.num_hidden)
        model.classifier = torch.nn.Linear(self.num_hidden, self.num_classes)
        # apply spectral normalization
        if self.spectral_normalization:
            replace_layers_rec(model)
        return model

    def activate_uncert_head(self, deactivate_backbone=True):
        """ """
        self.uncert_head = True
        return

    def deactivate_uncert_head(self):
        """ """
        self.uncert_head = False
        return

    def forward_hidden(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.pre_classifier(x)
        return x

    def forward(self, x):
        iterations = 1
        if self.uncert_head:
            iterations = 10
        last_layer_results = []
        for i in range(iterations):
            # x is flattened
            # extract explaining vars
            exp_vars = x[:, self.explaining_vars]
            # mask out explaining vars for resnet
            mask = torch.ones_like(x, dtype=torch.bool)
            mask[:, self.explaining_vars] = False
            intermed = x[mask]
            # reshape to image
            intermed = intermed.reshape(
                -1, self.image_shape[0], self.image_shape[1], self.image_shape[2]
            )

            # feed through resnet
            intermed = self.forward_hidden(intermed)
            # self.hidden = intermed

            # Dropout before classifier
            intermed = F.dropout(intermed, p=0.2, training=True)
            last_layer_results.append(intermed)
        last_layer_results = torch.stack(last_layer_results)
        x = last_layer_results.mean(axis=0)
        self.hidden = x
        self.uncertainty = last_layer_results.var(axis=0)
        return self.backbone.classifier(x)

    def get_uncertainty(self, dl, device, return_all=False):
        uncertainties = []
        for data, target in dl:
            data, target = data.to(device), target.to(device)
            output = self(data)
            uncertainties.append(self.uncertainty)
        uncertainties = torch.stack(uncertainties)
        if return_all:
            return uncertainties
        return uncertainties.mean()


class EfficientNetEnsemble(nn.Module, EinetUtils):
    """
    Wrapper for an Ensemble of EfficientNets.
    Train N EfficientNetDet first, then use
    this class to load the pretrained models.
    Make sure to use the same parameters.
    """

    def __init__(
        self,
        num_classes,
        image_shape,  # (C, H, W)
        explaining_vars=[],  # indices of variables that should be explained
        spec_norm_bound=0.9,
        num_hidden=32,
        spectral_normalization=True,
        ensemble_paths=[],
        map_location=None,
        **kwargs,
    ):
        super(EfficientNetEnsemble, self).__init__()
        self.num_classes = num_classes
        self.explaining_vars = explaining_vars
        self.image_shape = image_shape
        self.marginalized_scopes = None
        self.members = []
        self.num_hidden = num_hidden
        for path in ensemble_paths:
            member = EfficientNetDet(
                num_classes,
                image_shape,
                explaining_vars,
                spec_norm_bound,
                num_hidden,
                spectral_normalization,
            )
            member.load_state_dict(torch.load(path, map_location=map_location))
            self.members.append(member)

    def compute_normalization_values(self, dl, device):
        pass

    def activate_uncert_head(self, deactivate_backbone=True):
        pass

    def deactivate_uncert_head(self):
        pass

    def forward(self, x):
        """
        Forward pass through the ensemble.
        Simply averages the results of all members.
        """
        results = []
        for member in self.members:
            results.append(member(x))
        results = torch.stack(results)
        results = results.mean(axis=0)
        return results
