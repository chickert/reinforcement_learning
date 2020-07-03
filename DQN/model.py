# Model under construction
import torch
import torch.nn as nn


# TODO: deepQnetwork (class nn.module?) --
#   FORWARD: takes in state and outputs q_vals for actions
#   LOSS?
#   needs to be able to take in many? or just one? ie How to batch?

# TODO: agent (class) -- has agency (duh); interacts with environment and learns from it
#   takes step based on q_vals returned by model, or randomly by epsilon
#   TRAINs models based on replay buffer and update schedule

# TODO: replay memory (class)
#   initialized to some fixed capacity
#   can FILL with experiences, but not overfill
#   can SAMPLE from

# TODO: epsilon
#   anneals by some schedule