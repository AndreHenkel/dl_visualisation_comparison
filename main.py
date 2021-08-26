    #AlexNet and processing code taken from: https://pytorch.org/hub/pytorch_vision_alexnet/

import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
import utils


from dataloader import DataLoader
from aopc import AOPC

from lrp_classifier import LRP_Classifier
from deconv_classifier import Deconv_Classifier
from sensitivity_classifier import Sensitivity_Classifier


print("Cuda is available: ", torch.cuda.is_available())

perturbation_steps = 100
perturbation_area = (3,9,9)
image_count = 10

#load pretrained alex net into
alexnet = torch.hub.load('pytorch/vision:v0.9.0','alexnet',pretrained=True) #load pretrained alexnet as used in Samek et. al

#init relevance models
lrp_model = LRP_Classifier()
lrp_model.load_state_dict(alexnet.state_dict())
deconv_model = Deconv_Classifier()
sensitivity_model = Sensitivity_Classifier()

#deactivate i.e. Dropout etc.
alexnet.eval()
lrp_model.eval()
deconv_model.eval()
sensitivity_model.eval()


imagenette_2_320px_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"

dl = DataLoader(".",imagenette_2_320px_url, image_count)
image_batch,image_idx = dl.get_image_batch()

#print(image_idx)
#load to cuda
if(torch.cuda.is_available() == True):
    lrp_model.cuda()
    alexnet.cuda()
    deconv_model.cuda()
    sensitivity_model.cuda()
    image_batch = image_batch.to(device='cuda')

aopc = AOPC(image_batch, image_idx, lrp_model, alexnet, perturbation_steps=perturbation_steps, perturbation_area=perturbation_area)
aopc_deconv = AOPC(image_batch, image_idx, deconv_model, alexnet, perturbation_steps=perturbation_steps, perturbation_area=perturbation_area)
aopc_sensitivity = AOPC(image_batch, image_idx, sensitivity_model, alexnet, perturbation_steps=perturbation_steps, perturbation_area=perturbation_area)


print("------------------------------------------------")
print("Calculating AOPC curve relative to random with: ")
print("perturbation_steps:      ", perturbation_steps)
print("perturbation_area:       ", perturbation_area)
print("image_count:             ", image_count)
print("------------------------------------------------")


#aopc data
#l2 or l_inf norm
#sensitivity
aopc_data_sensitivity_l2 = aopc_sensitivity.get_aopc_curve(rule="sensitivity",norm="l2")
aopc_data_sensitivity_l_inf = aopc_sensitivity.get_aopc_curve(rule="sensitivity",norm="l_inf")
#deconv
aopc_data_deconv_l2 = aopc_deconv.get_aopc_curve(rule="deconv",norm="l2")
aopc_data_deconv_l_inf = aopc_deconv.get_aopc_curve(rule="deconv",norm="l_inf")
#lrp
aopc_data_eps_0 = aopc.get_aopc_curve(rule="epsilon0")
aopc_data_eps_0_01 = aopc.get_aopc_curve(rule="epsilon0_01")
aopc_data_eps_100 = aopc.get_aopc_curve(rule="epsilon100")
#random
aopc_data_random = aopc.compare_to_random()


#graph of aopc relative to random baseline
#sensitivity
rel_graph_sensitivity_l2 = utils.relative_graph(aopc_data_random,aopc_data_sensitivity_l2 )
rel_graph_sensitivity_l_inf = utils.relative_graph(aopc_data_random,aopc_data_sensitivity_l_inf )
#deconv
rel_graph_deconv_l2 = utils.relative_graph(aopc_data_random,aopc_data_deconv_l2 )
rel_graph_deconv_l_inf = utils.relative_graph(aopc_data_random,aopc_data_deconv_l_inf )
#lrp
rel_graph_eps_0 = utils.relative_graph(aopc_data_random,aopc_data_eps_0 )
rel_graph_eps_0_01 = utils.relative_graph(aopc_data_random,aopc_data_eps_0_01 )
rel_graph_eps_100 = utils.relative_graph(aopc_data_random,aopc_data_eps_100 )


#plot graphs
plt.plot(rel_graph_sensitivity_l2[0].detach().numpy(), label="sensitivity_l2")
plt.plot(rel_graph_sensitivity_l_inf[0].detach().numpy(), label="sensitivity_l_inf")

plt.plot(rel_graph_deconv_l2[0].detach().numpy(), label="deconv_l2")
plt.plot(rel_graph_deconv_l_inf[0].detach().numpy(), label="deconv_l_inf")

plt.plot(rel_graph_eps_0[0].detach().numpy(), label="eps_0")
plt.plot(rel_graph_eps_0_01[0].detach().numpy(), label="eps_0.01")
plt.plot(rel_graph_eps_100[0].detach().numpy(), label="eps_100")



#plot
plt.xlabel('perturbation steps')
plt.ylabel('AOPC relative to random')

plt.legend()
plt.show()

#the image can then be manually saved via the graphical interface
