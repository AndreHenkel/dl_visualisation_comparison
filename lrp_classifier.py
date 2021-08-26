import torch
import torch.nn as nn

import lrp_framework.lrp as lrp

from utils import calculate_linear_lrp_fast

class LRP_Classifier(nn.Module):
    def __init__(self, num_classes=1000) -> None:
        super(LRP_Classifier,self).__init__()

        #just the same structure as AlexNet, for easy import
        #using lrp modules, so the AlexNet weights are then directly used by the lrp framework
        self.features = lrp.Sequential(
                lrp.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                lrp.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                lrp.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                lrp.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                lrp.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = lrp.Sequential(
                nn.Dropout(),
                lrp.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                lrp.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                lrp.Linear(4096, num_classes),
            )


    def forward(self, x: torch.Tensor, explain=False, rule="epsilon", pattern=None) -> torch.Tensor:
        #x = self.seq(x,explain,rule,pattern)
        x = self.features(x,explain,rule,pattern)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x,explain,rule,pattern)
        return x
