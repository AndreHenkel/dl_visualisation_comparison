from hm_deconv_classifier import HM_Deconv_Classifier
from dataloader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import utils
import torch

model = Deconv_Classifier()
model.eval() #important, because otherwise dropout layers would apply and it would be a nightmare... (hours of time...)


dl = DataLoader("/home/ajohe/git/imagenette2-320/train",1)
#img,img_idx = dl.get_image_batch() #minibatch of 1

img_name = "ILSVRC2012_val_00038005.JPEG"
image_path = "/home/ajohe/git/imagenette2-320/train/golf ball/"+img_name

img = utils.load_image(image_path)
img = utils.normalize_image(img)

img.requires_grad_(True)
img.grad = None
#pred = model(img)[0,img_idx] #prediction of label
pred = model(img)
label = pred.argmax(dim=1)
pred_score = pred[:,label]

print(pred_score)
pred_score.backward()


# pred.backward()

# pull the gradients out of the model
gradients = model.get_activations_gradient()

print("gradients.shape: ",gradients.shape)

# pool the gradients across the channels
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

print("pooled_gradients.shape: ",pooled_gradients.shape)

# get the activations of the last convolutional layer
activations = model.get_activations(img).detach()

print("act_shape: ",activations.shape)


# TODO: what does happen here???
# weight the channels by corresponding gradients (which have been mean'ed over all channels)
for i in range(256):
    activations[:, i, :, :] *= pooled_gradients[i]



# average the channels of the activations
heatmap = torch.mean(activations, dim=1).squeeze()

# relu on top of the heatmap
# expression (2) in https://arxiv.org/pdf/1610.02391.pdf
heatmap = np.maximum(heatmap, 0)

# normalize the heatmap
heatmap /= torch.max(heatmap)

#hm = heatmap.squeeze()
print(heatmap.shape)
print(heatmap)
#hm = hm.detach().numpy()
# draw the heatmap
plt.matshow(heatmap.squeeze())
plt.show()
#show in image
import cv2


def hm_to_rel(hm_img):
    """
    heatmap img to relevance
    """


heatmap = heatmap.detach().numpy()

#img
img = cv2.imread(image_path)
# img = img.squeeze()
# img = img.permute(1,2,0)
# img = img.detach().numpy()


heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#heatmap = cv2.resize(heatmap, (12, 12))
heatmap = np.uint8(255 * heatmap)
print(heatmap)
print(heatmap.shape)
#heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#superimposed_img = heatmap * 0.4 + img
print(heatmap)
print(heatmap.shape)
#superimposed_img  = np.sum(superimposed_img, axis=2)
#print(superimposed_img.shape)
plt.imshow(heatmap,cmap='gray')
plt.show()
# cv2.imwrite('./map.jpg', heatmap)
