import numpy as np
import torch
from torch.utils import data

# dataset = np.load('./expert.npz')
# tensor_dataset = data.TensorDataset(torch.Tensor(dataset['obs']), torch.Tensor(dataset['action']))
# dataloader = data.DataLoader(tensor_dataset, batch_size=50, shuffle=True)

dataset = np.load('./expert.npz')
tensor_dataset = data.TensorDataset(torch.Tensor(dataset['obs']), torch.Tensor(dataset['action']))

train_size = int(0.8 * len(tensor_dataset))
test_size = len(tensor_dataset) - train_size
train_dataset, test_dataset = data.random_split(tensor_dataset, [train_size, test_size])

train_dataloader = data.DataLoader(train_dataset, batch_size=50, shuffle=True)
test_dataloader = data.DataLoader(test_dataset, batch_size=50, shuffle=True)

