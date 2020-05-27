# import logging
# # Create and configure logger
# # logging.basicConfig(filename="newfile.log",
# #                     format='%(asctime)s %(message)s',
# #                     filemode='w')
# logging.basicConfig()
# Creating an object
# logger = logging.getLogger()
#
import torch

x = torch.tensor([8, 9])
print(x.shape)

if x.shape == torch.Size([2]):
    print("hi")
    x = torch.unsqueeze(x, 0)

print(x.shape)