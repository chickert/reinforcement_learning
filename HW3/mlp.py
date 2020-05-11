import torch.nn as nn
import torch.nn.functional as f
import torch


class FullyConnectedLayer(nn.Module):
    def __init__(self, in_features, out_features, activation):
        super(FullyConnectedLayer, self).__init__()
        self.weight_matrix = nn.Linear(in_features, out_features, bias=False)
        self.activation = activation

    def forward(self, input, activation):
        output = self.weight_matrix(input)
        if self.activation:
            output = self.activation(output)
        return output


class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_features, hidden_layer_sizes, activation):
        super(MultiLayerPerceptron, self).__init__()
        self.num_hidden_layers = len(hidden_layer_sizes)
        self.network = nn.Sequential(
            *[
                FullyConnectedLayer(
                    in_features=hidden_layer_sizes[i - 1] if i > 0 else in_features,
                    out_features=hidden_layer_sizes[i],
                    activation=activation if i < self.num_hidden_layers else None
                )
                for i in range(self.num_hidden_layers)
            ]
        )


    def forward(self, input):
        return self.network(input)