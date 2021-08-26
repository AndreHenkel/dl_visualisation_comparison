import torch

import numpy as np
import torch.nn as nn
import torchvision
import utils
import matplotlib.pyplot as plt
from own_deconv_classifier import Own_Deconv_Classifier


model = Own_Deconv_Classifier()

img_name = "ILSVRC2012_val_00002314.JPEG"
folder_name = "golf ball/"
image_path = "/home/ajohe/git/imagenette2-320/train/"+folder_name+img_name

img = utils.load_image(image_path)
img = utils.normalize_image(img)

pred = model(img)

idx = pred[0].argmax()
pred_score = pred[0][idx]

pred_score.backward()

R = model.calc_heat()


#print(R)
print(R.shape)
print(torch.max(R))
print(torch.min(R))
print(pred_score)

R = R.squeeze()
R = R.permute(1,2,0)
R = R.cpu().detach().numpy()

print(R.shape)
R = R.sum(axis=2)
#R = R[:,:,2]

print(R.shape)
R = utils.project(R,(0,255))
#
print(np.mean(R))
#
# R[R < 255.0] = 0.0

#plt.imshow(R)
plt.imshow(R,cmap='gray')
plt.show()
