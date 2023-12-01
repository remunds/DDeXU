import torch.nn as nn
from EinsumNetwork import Graph, EinsumNetwork


class ResNetHidden(nn.Module):
    """
    ResNet model
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_layers=3,
        num_hidden=128,
        dropout_rate=0.1,
        **classifier_kwargs
    ):
        super(ResNetHidden, self).__init__()
        # Defines class meta data.
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.classifier_kwargs = classifier_kwargs
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Defines the hidden layers.
        self.input_layer = nn.Linear(self.input_dim, self.num_hidden)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.dense_layers = [self.make_dense_layer() for _ in range(num_layers)]
        self.dense_layers = nn.ModuleList(self.dense_layers)

        # Defines the output layer.
        self.classifier = self.make_output_layer()

    def forward(self, inputs):
        # Projects the 2d input data to high dimension.
        hidden = self.input_layer(inputs)

        # Computes the ResNet hidden representations.
        for i in range(self.num_layers):
            resid = self.dense_layers[i](hidden)
            resid = self.dropout(resid)
            hidden = hidden + resid

        return self.classifier(hidden)

    def make_dense_layer(self):
        """Uses the Dense layer as the hidden layer."""
        return nn.Sequential(
            nn.Linear(self.num_hidden, self.num_hidden), self.activation
        )

    def make_output_layer(self):
        """Uses the Dense layer as the output layer."""
        return nn.Linear(self.num_hidden, self.output_dim, **self.classifier_kwargs)


class ResNetSPN(ResNetHidden):
    #     def __init__(self, spec_norm_bound=0.9, **kwargs):
    #         self.spec_norm_bound = spec_norm_bound
    #         super().__init__(**kwargs)

    # def make_dense_layer(self):
    #     """applies spectral normalization to the hidden layer."""
    #     dense = nn.Linear(self.num_hidden, self.num_hidden)
    #     # TODO: this is different to tf, since it does not use the spec_norm_bound...
    #     return nn.Sequential(
    #         nn.utils.parametrizations.spectral_norm(dense), self.activation
    #     )

    def make_output_layer(self):
        """uses einet as the output layer."""
        return nn.Linear(self.num_hidden, self.output_dim, **self.classifier_kwargs)

    def forward_latent(self, inputs):
        # Projects the 2d input data to high dimension.
        hidden = self.input_layer(inputs)

        # Computes the ResNet hidden representations.
        for i in range(self.num_layers):
            resid = self.dense_layers[i](hidden)
            resid = self.dropout(resid)
            hidden = hidden + resid

        return hidden

    def replace_output_layer(self, device):
        """uses einet as the output layer."""

        self.max_num_epochs = 5
        self.batch_size = 100

        depth = 3
        num_repetitions = 20
        K = 10
        online_em_frequency = 1
        online_em_stepsize = 0.05
        exponential_family = EinsumNetwork.NormalArray
        exponential_family_args = {"min_var": 1e-6, "max_var": 0.1}

        self.graph = Graph.random_binary_trees(
            num_var=self.num_hidden, depth=depth, num_repetitions=num_repetitions
        )

        args = EinsumNetwork.Args(
            num_var=self.num_hidden,
            num_dims=1,
            num_classes=self.output_dim,
            num_sums=K,
            num_input_distributions=K,
            exponential_family=exponential_family,
            exponential_family_args=exponential_family_args,
            online_em_frequency=online_em_frequency,
            online_em_stepsize=online_em_stepsize,
        )

        einet = EinsumNetwork.EinsumNetwork(self.graph, args)
        einet.initialize()
        einet.to(device)
        self.classifier = einet


class ResNetSPNEnd2End(ResNetHidden):
    def __init__(self, spec_norm_bound=0.9, **kwargs):
        self.spec_norm_bound = spec_norm_bound
        super().__init__(**kwargs)

    def make_dense_layer(self):
        """applies spectral normalization to the hidden layer."""
        dense = nn.linear(self.num_hidden, self.num_hidden)
        # todo: this is different to tf, since it does not use the spec_norm_bound...
        return nn.sequential(
            nn.utils.parametrizations.spectral_norm(dense), self.activation
        )

    def make_output_layer(self):
        """uses einet as the output layer."""

        self.max_num_epochs = 5
        self.batch_size = 100

        depth = 3
        num_repetitions = 20
        K = 10
        online_em_frequency = 1
        online_em_stepsize = 0.05
        exponential_family = EinsumNetwork.NormalArray
        exponential_family_args = {"min_var": 1e-6, "max_var": 0.1}

        self.graph = Graph.random_binary_trees(
            num_var=self.num_hidden, depth=depth, num_repetitions=num_repetitions
        )

        args = EinsumNetwork.Args(
            num_var=self.num_hidden,
            num_dims=1,
            num_classes=self.output_dim,
            num_sums=K,
            num_input_distributions=K,
            exponential_family=exponential_family,
            exponential_family_args=exponential_family_args,
            online_em_frequency=online_em_frequency,
            online_em_stepsize=online_em_stepsize,
        )

        einet = EinsumNetwork.EinsumNetwork(self.graph, args)
        einet.initialize()
        return einet
