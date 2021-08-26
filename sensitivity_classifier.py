#model for sensitivity analysis
#inspiration taken from: https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/vanilla_backprop.py

import torch
import torch.nn as nn


class Sensitivity_Classifier(nn.Module):
    def __init__(self, num_classes=1000) -> None:
        super(Sensitivity_Classifier,self).__init__()

        self.alexnet = torch.hub.load('pytorch/vision:v0.9.0','alexnet',pretrained=True) #load pretrained alexnet as used in Samek et. al

        self.features = self.alexnet.features
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = self.alexnet.classifier

        self.gradients = None
        self.eval()


    def act_hook_func(self,grad):
        self.gradients = grad

    def hook_function(self,module,input,output):
        #print("hooked")
        self.gradients = input[0]
        #print(module,input,output)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.register_hook(self.act_hook_func)
        #self.features[0].register_backward_hook(self.hook_function)
        x = self.features[0](x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2)(x)

        x = self.features[3](x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2)(x)

        x = self.features[6](x)
        x = nn.ReLU(inplace=True)(x)

        x = self.features[8](x)
        x = nn.ReLU(inplace=True)(x)

        x = self.features[10](x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2)(x)

        #keeps shape
        x = self.avgpool(x) #here avg pool. in Samek et. al paper max pool was used

        x = torch.flatten(x, 1)

        #classifier part (dropout can be neglegted, since we won't train the network anymore)
        #x = self.dropout(x)
        x = self.classifier[1](x)
        x = nn.ReLU(inplace=True)(x)

        #x = self.dropout(x)
        x = self.classifier[4](x)
        x = nn.ReLU(inplace=True)(x)

        x = self.classifier[6](x)
        return x
