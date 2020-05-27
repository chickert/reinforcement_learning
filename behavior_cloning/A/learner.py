import torch
import torch.nn as nn
import torch.nn.functional as f

from mlp import MultiLayerPerceptron


class BehaviorCloningModel(nn.Module):
    """
    Model predicts action given state of the environment
    """
    def __init__(self, num_envstate_dims, num_action_dims,
                 hidden_layer_sizes, activation=f.selu, seed=0):
        torch.manual_seed(seed)
        super(BehaviorCloningModel, self).__init__()
        self.mlp = MultiLayerPerceptron(
            in_features=num_envstate_dims,
            hidden_layer_sizes=hidden_layer_sizes + [num_action_dims],
            activation=activation)

    def forward(self, state):
        input = state
        return self.mlp(input)

