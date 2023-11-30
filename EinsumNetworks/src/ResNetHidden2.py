import torch.nn as nn

class ResNetHidden2(nn.Module):
    """
    ResNet model with ability to return latent space
    """
    def __init__(self, num_classes, input_dim, num_layers=3, num_hidden=128, dropout_rate=0.1, **classifier_kwargs):
        super(ResNetHidden2, self).__init__()
        # Defines class meta data.
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.classifier_kwargs = classifier_kwargs
        self.input_dim = input_dim

        # Defines the hidden layers.
        self.input_layer = nn.Linear(self.input_dim, self.num_hidden)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.dense_layers = [self.make_dense_layer() for _ in range(num_layers)]
        self.dense_layers = nn.ModuleList(self.dense_layers)

        # Defines the output layer.
        self.classifier = self.make_output_layer(num_classes)

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
            nn.Linear(self.num_hidden, self.num_hidden),
            self.activation
        )

    def make_output_layer(self, num_classes):
        """Uses the Dense layer as the output layer."""
        return nn.Linear(self.num_hidden, num_classes, **self.classifier_kwargs)

class ResNetSPN(ResNetHidden2):
    def __init__(self, spec_norm_bound=0.9, **kwargs):
        self.spec_norm_bound = spec_norm_bound
        super().__init__(**kwargs)

    def make_dense_layer(self):
        """Applies spectral normalization to the hidden layer."""
        dense = nn.Linear(self.num_hidden, self.num_hidden)
        # TODO: this is different to TF, since it does not use the spec_norm_bound...
        return nn.Sequential(
            nn.utils.parametrizations.spectral_norm(dense),
            self.activation
        )

    def make_output_layer(self, num_classes):
        """Uses Gaussian process as the output layer."""
        # TODO: Use SPN here
        return super().make_output_layer(num_classes)
