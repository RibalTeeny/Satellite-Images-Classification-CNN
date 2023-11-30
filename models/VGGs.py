import torch
import numpy as np
from torchvision.models import vgg19, vgg16
from torchvision.models.vgg import VGG19_Weights, VGG16_Weights

class VGG(torch.nn.Module):
    def __init__(self, layers=19):
        super(VGG, self).__init__()

        if layers == 19:
            # pretrained VGG19
            self.vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).eval()

            # network till the last conv layer (35th)
            print("len of features:", len(self.vgg.features))
            print(self.vgg.features)
            self.features_conv = self.vgg.features[:-1] # was 36
            print("len of features after cut:", len(self.features_conv))
        
        elif layers == 16:
            # pretrained VGG16
            self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

            # network till the last conv layer (35th)
            self.features_conv = self.vgg.features[:-1]

        # the max-pool in VGG19 that is after the last conv layer
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2,
                                            stride=2,
                                            padding=0,
                                            dilation=1,
                                            ceil_mode=False)

        # vgg's classifier
        self.classifier = self.vgg.classifier
        print(self.classifier)

        # extracted gradients
        self.gradients = None

    # hook
    def activation_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        # VGG 19 till the last conv layer
        x = self.features_conv(x)
        # register the hook in the forward pass
        hook = x.register_hook(self.activation_hook)

        # continue finishing the VGG19
        x = self.max_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)

        return x

    # extract gradient
    def get_activation_gradient(self):
        return self.gradients

    # extract the activation after the last ReLU
    def get_activation(self, x):
        return self.features_conv(x)