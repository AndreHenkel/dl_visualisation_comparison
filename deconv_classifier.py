import torch
import torch.nn as nn


class Deconv_Classifier(nn.Module):
    def __init__(self, num_classes=1000) -> None:
        super(Deconv_Classifier,self).__init__()
        self.img_shape = None

        self.feature_out_act = None

        #self.feature_output = torch.tensor((256*6*6))
        self.max_pool1_indices = None
        self.max_pool2_indices = None
        self.max_pool3_indices = None


        self.unpool1 = nn.MaxUnpool2d(kernel_size=3, stride=2)
        self.unpool2 = nn.MaxUnpool2d(kernel_size=3, stride=2)
        self.unpool3 = nn.MaxUnpool2d(kernel_size=3, stride=2)

        self.alexnet = torch.hub.load('pytorch/vision:v0.9.0','alexnet',pretrained=True) #load pretrained alexnet as used in Samek et. al

        self.features = self.alexnet.features
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = self.alexnet.classifier

        self.convT1 = nn.ConvTranspose2d(64,3,kernel_size=11,stride=4,padding=2,bias=False)
        self.convT1.weight = self.features[0].weight
        #self.convT1.bias = self.features[0].bias
        self.convT2 = nn.ConvTranspose2d(192,64,kernel_size=5,padding=2,bias=False)
        self.convT2.weight = self.features[3].weight
        #self.convT2.bias = self.features[3].bias
        self.convT3 = nn.ConvTranspose2d(384,192,kernel_size=3,padding=1,bias=False)
        self.convT3.weight = self.features[6].weight
        #self.convT3.bias = self.features[6].bias
        self.convT4 = nn.ConvTranspose2d(256,384,kernel_size=3,padding=1,bias=False)
        self.convT4.weight = self.features[8].weight
        #self.convT4.bias = self.features[8].bias
        self.convT5 = nn.ConvTranspose2d(256,256,kernel_size=3,padding=1,bias=False)
        self.convT5.weight = self.features[10].weight
        #self.convT5.bias = self.features[10].bias

        #always stores last pass through's gradients
        self.feature_out_gradients = None


    def activations_hook(self,grad):
        self.feature_out_gradients = grad
        #print("Stored gradients")

#check https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py for order of deconv steps
    def calc_heat(self):
        #start with feature_out_act #before last avg pool
        if self.feature_out_gradients == None:
            print("You have to use .backward() first")
            return

        #R = torch.matmul(self.feature_out_act, self.feature_out_gradients)

        # pooled_gradients = torch.mean(self.feature_out_gradients, dim=[0, 2, 3])
        # for i in range(256):
        #     self.feature_out_act[:, i, :, :] *= pooled_gradients[i]


        #(batch, 256, 6, 6)
        #print("feat_out_act:",self.feature_out_act.shape)
        R = self.feature_out_act
        #R = self.feature_out_gradients

        #5
        R = self.unpool3(R,self.max_pool3_indices)
        R = nn.ReLU()(R)
        R = self.convT5(R)

        #4
        R = nn.ReLU()(R)
        R = self.convT4(R)

        #3
        R = nn.ReLU()(R)
        R = self.convT3(R)

        #2
        R = self.unpool2(R,self.max_pool2_indices)
        R = nn.ReLU()(R)
        R = self.convT2(R)

        #1
        R = self.unpool1(R,self.max_pool1_indices)
        R = nn.ReLU()(R)
        R = self.convT1(R,output_size=self.image_shape)

        return R



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.image_shape = x.shape
        x = self.features[0](x)
        x = nn.ReLU(inplace=True)(x)
        x,self.max_pool1_indices = nn.MaxPool2d(kernel_size=3, stride=2,return_indices=True)(x)

        x = self.features[3](x)
        x = nn.ReLU(inplace=True)(x)
        x,self.max_pool2_indices = nn.MaxPool2d(kernel_size=3, stride=2,return_indices=True)(x)

        x = self.features[6](x)
        x = nn.ReLU(inplace=True)(x)

        x = self.features[8](x)
        x = nn.ReLU(inplace=True)(x)

        x = self.features[10](x)
        x = nn.ReLU(inplace=True)(x)
        x,self.max_pool3_indices = nn.MaxPool2d(kernel_size=3, stride=2,return_indices=True)(x)


        #print(x.shape)
        #hook
        #stores the gradients in the backward pass
        h = x.register_hook(self.activations_hook)
        self.feature_out_act = x
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
