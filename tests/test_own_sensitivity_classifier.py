import torch

import numpy as np
import torch.nn as nn
import torchvision
import utils
import matplotlib.pyplot as plt
from own_sensitivity_classifier import Own_Sensitivity_Classifier


model = Own_Sensitivity_Classifier()

img_name = "ILSVRC2012_val_00002314.JPEG"
folder_name = "golf ball/"
image_path = "/home/ajohe/git/imagenette2-320/train/"+folder_name+img_name

img = utils.load_image(image_path)
img = utils.normalize_image(img)

img.requires_grad_(True)

pred = model(img)

idx = pred[0].argmax()
pred_score = pred[0][idx]

model.zero_grad()

pred_score.backward()

R = model.gradients#.data.numpy()[0]
#R = model.get_out_gradient()
print(R.shape)
#

R = R.squeeze()
R = R.permute(1,2,0)
R = R.cpu().detach().numpy()
print(R.shape)

R = R.sum(axis=2)
#R = R[:,:,2]

print(R.shape)
R = utils.project(R,(0,255))

#R [R< 254.0] = 0.0

plt.imshow(R,cmap='gray')
plt.show()
