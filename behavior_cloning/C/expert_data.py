import numpy as np
import torch
from torch.utils import data

dataset = np.load('./expert.npz')
tensor_dataset = data.TensorDataset(torch.Tensor(dataset['obs']), torch.Tensor(dataset['action']))
dataloader = data.DataLoader(tensor_dataset, batch_size=50, shuffle=True)


