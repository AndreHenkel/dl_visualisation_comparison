import torch
import lrp
from PIL import Image
from utils import relevance_map_to_ordered_sequence,create_random_sorted_r_seq, load_image, MoRF, R_pix

import matplotlib.pyplot as plt

import numpy as np

from dataloader import DataLoader


dl = DataLoader("/home/ajohe/git/imagenette2-320",10)


b = dl.get_image_batch()
# print(b.shape)
# print(torch.mean(b))
#
# data_loader = dl.get_data_loader()
#
# dl_it = iter(data_loader)
# for i in range(11):
#     x,y = next(dl_it)
#     print(x.shape,y)
#     print(torch.mean(x))

# for i in range(10):
#     batch,label = dl.get_image_batch()
#     print(batch.shape)
#     print(label)
#     print(torch.mean(batch))
#
# a = torch.zeros((3,3))
# b = torch.ones((3,3))
#
# d = torch.ones((3,3))
#
# c = torch.stack((a,b))
# c = torch.stack((c,d),dim=1)
# print(c.shape)


# a = [[1,2,3,4], [4,5,6,7], [0,9,9,9]]
#
# print(a)
# print(torch.Tensor(a))

# a = zip([1,2,3,4], [6,7,8,9])
# b = [9,9,9,9]
# c = zip(*zip(*a), b)
#
#
# for x,y,z in c:
#     print(x,y,z)

# a = -2
#
# a = max(0,min(224,a))
#
# print(a)


# a = torch.zeros((9,9))
# print(a)
# b = torch.ones((3,3))
# print(b)
#
# a[0:3,0:3] = b
# print(a)

#a = create_random_sorted_r_map((224,224))

#leatherback_turtle
#wine_bottle
#wooden_spoon
# a = load_image("../images/wooden_spoon.jpg")
#
# plt.imshow(a.permute(1,2,0))
# # plt.show()
#
# x,y = np.random.randint(0,224,size=(2))
#
# b = MoRF(a,R_pix(x,y,10),area_size=(3,25,25))
# plt.imshow(b.permute(1,2,0))
# plt.show()

# print(torch.max(a))
# print(torch.min(a))
# print(torch.mean(a))
# model = Sequential(
#     lrp.Conv2d(1, 32, 3, 1, 1),
#     torch.nn.ReLU(),
#     torch.nn.MaxPool2d(2, 2),
#     torch.nn.Flatten(),
#     lrp.Linear(14*14*32, 10)
# )
#
#
#
# x = ... # business as usual
# y_hat = model.forward(x, explain=True, rule="alpha2beta1")
# y_hat = y_hat[torch.arange(batch_size), y_hat.max(1)[1]] # Choose maximizing output neuron
# y_hat = y_hat.sum()
#
# # Backward pass (do explanation)
# y_hat.backward()
# explanation = x.grad
