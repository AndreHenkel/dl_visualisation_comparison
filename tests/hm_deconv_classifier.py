#this is the AlexNet Classifier with integrated Deconv functionalities
#inspiration taken from https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82

import torch
import torch.nn as nn
import torchvision
import numpy as np

class HM_Deconv_Classifier(nn.Module):
    def __init__(self):
        super(HM_Deconv_Classifier,self).__init__()

        self.alexnet = torch.hub.load('pytorch/vision:v0.9.0','alexnet',pretrained=True) #load pretrained alexnet as used in Samek et. al

        self.features = self.alexnet.features
        self.classifier = self.alexnet.classifier
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        #always stores last pass through's gradients
        self.gradients = None


    def activations_hook(self,grad):
        self.gradients = grad

    def forward(self,x):
        x = self.features(x)

        #hook
        h = x.register_hook(self.activations_hook)

        #pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    #returns the hooked gradients of the last forward pass after the features part(?)
    def get_activations_gradient(self):
        return self.gradients

    #returns the activation after the conv layers (features)
    def get_activations(self,x):
        return self.features(x)
