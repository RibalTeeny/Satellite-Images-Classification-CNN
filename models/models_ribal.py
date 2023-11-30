import torch.nn as nn
import torch
from typing import Any, cast, Dict, List, Optional, Union, TypeVar
from torchvision.models._api import WeightsEnum
from torchvision.models.vgg import VGG19_Weights




class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

def vgg19(*, weights: Optional[VGG19_Weights] = None, progress: bool = True, **kwargs: Any) -> VGG:
    """VGG-19 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        weights (:class:`~torchvision.models.VGG19_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG19_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.VGG19_Weights
        :members:
    """
    weights = VGG19_Weights.verify(weights)

    return _vgg("E", False, weights, progress, **kwargs)

def _vgg(cfg: str, batch_norm: bool, weights: Optional[WeightsEnum], progress: bool, **kwargs: Any) -> VGG:
    if weights is not None:
        kwargs["init_weights"] = False
        if weights.meta["categories"] is not None:
            _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
    return model

V = TypeVar("V")

def _ovewrite_named_param(kwargs: Dict[str, Any], param: str, new_value: V) -> None:
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead.")
    else:
        kwargs[param] = new_value

# import torch
# import torch.nn as nn
# import torchvision.models as models
# import pytorch_lightning as pl

# number of layers in each block
# VGGs = {"VGG11": [1,1,2,2,2], # 1x64, 1x128, 2x256, 2x512, 2x512
#         "VGG13": [2,2,2,2,2], #
#         "VGG16": [2,2,3,3,3], #
#         "VGG19": [2,2,4,4,4], #
#         }

# class VGG16Lightning(pl.LightningModule):
#     def __init__(self, num_classes=1):
#         super().__init__()
#         self.save_hyperparameters()  # usefull for loading from checkpoint, the init params will be preserved
#         self.vgg16 = models.vgg16(pretrained=True)
#         for param in self.vgg16.features.parameters():
#             param.requires_grad = False
#         num_ftrs = self.vgg16.classifier[6].in_features
#         self.vgg16.classifier[6] = nn.Linear(num_ftrs, num_classes)

#     def forward(self, x):
#         return self.vgg16(x)

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         loss = nn.functional.cross_entropy(y_hat, y)
#         self.log('train_loss', loss)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         loss = nn.functional.cross_entropy(y_hat, y)
#         self.log('val_loss', loss)

#     def configure_optimizers(self):
#         optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
#         #optimizer = torch.optim.Adam(self.parameters(), lr=0.00001)
#         return {"optimizer": optimizer}
    
#     def get_model_name(self):
#         return "VGG16"
    
# class Block(nn.Module):
#     def __init__(self, in_channels, out_channels, no_layers) -> None:
#         super().__init__()
#         self.relu - nn.ReLU()
#         layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1), self.relu]

#         for _ in no_layers:
#             layers.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
#             layers.append(self.relu)
        
#         layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

#         self.conv_block = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.conv_block(x)
    
# class VGG(nn.Module):
#     def __init__(self, in_channels, classes, VGG_type) -> None:
#         super().__init__()
#         self.feature_maps = [in_channels, 64, 128, 256, 512, 512]
#         self.convs = nn.ModuleList([])


#         for i in range(len(VGG_type)):
#             self.convs.append(Block(self.feature_maps[i], self.feature_maps[i+1], VGG_type[i]))

        
#         self.fc1 = nn.Linear(7*7*512, 4096)
#         self.fc2 = nn.Linear(4096, 4096)
#         self.fc3 = nn.Linear(4096, 1000)


#         self.dropout = nn.Dropout(0.5)
#         self.relu = nn.ReLU()

#         self.init_weight()


#     def forward(self, x):
#         for block in self.convs:
#             x = block(x)
        
#         x = torch.flatten(x, 1)
#         x = self.relu(self.dropout(self.fc1(x)))
#         x = self.relu(self.dropout(self.fc2(x)))
#         x = self.fc3(x)

#         return x
    
#     def init_weight(self):
#         for layer in self.modules():
#             if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
#                 nn.init.normal_(layer.weights, std=0.01)
#                 nn.init.constant_(layer.bias, 0)


# class VGG16(nn.Module):
#     def __init__(self, num_classes=2, pretrained=True):
#         super(VGG16, self).__init__()
#         self.features = models.vgg16(pretrained=pretrained).features
#         self.classifier = nn.Sequential(
#             nn.Linear(25088, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, num_classes)
#         )
    
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x

# # Define VGG19 model
# class VGG19(nn.Module):
#     def __init__(self, num_classes=2, pretrained=True):
#         super(VGG19, self).__init__()
#         self.features = models.vgg19(pretrained=pretrained).features
#         self.classifier = nn.Sequential(
#             nn.Linear(25088, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, num_classes)
#         )
    
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x

# Define VGG16 LightningModule
# class VGG16(pl.LightningModule):
#     def __init__(self, num_classes=2):
#         super(VGG16, self).__init__()
#         self.features = models.vgg16(pretrained=True).features
#         self.classifier = nn.Sequential(
#             nn.Linear(25088, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, num_classes)
#         )
#         self.loss_fn = nn.CrossEntropyLoss()
    
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x
    
#     def training_step(self, batch, batch_idx):
#         images, labels = batch
#         outputs = self(images)
#         loss = self.loss_fn(outputs, labels)
#         self.log("train_loss", loss)
#         return loss
    
#     def configure_optimizers(self):
#         optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
#         return optimizer
# import torchmetrics

# class PyTorchVGG16(nn.Module):

#     def __init__(self, num_classes, pretrained=True):
#         super().__init__()
        
#         # calculate same padding:
#         # (w - k + 2*p)/s + 1 = o
#         # => p = (s(o-1) - w + k)/2
        
#         self.block_1 = nn.Sequential(
#                 nn.Conv2d(in_channels=3,
#                           out_channels=64,
#                           kernel_size=(3, 3),
#                           stride=(1, 1),
#                           # (1(32-1)- 32 + 3)/2 = 1
#                           padding=1), 
#                 nn.ReLU(),
#                 nn.Conv2d(in_channels=64,
#                           out_channels=64,
#                           kernel_size=(3, 3),
#                           stride=(1, 1),
#                           padding=1),
#                 nn.ReLU(),
#                 nn.MaxPool2d(kernel_size=(2, 2),
#                              stride=(2, 2))
#         )
        
#         self.block_2 = nn.Sequential(
#                 nn.Conv2d(in_channels=64,
#                           out_channels=128,
#                           kernel_size=(3, 3),
#                           stride=(1, 1),
#                           padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(in_channels=128,
#                           out_channels=128,
#                           kernel_size=(3, 3),
#                           stride=(1, 1),
#                           padding=1),
#                 nn.ReLU(),
#                 nn.MaxPool2d(kernel_size=(2, 2),
#                              stride=(2, 2))
#         )
        
#         self.block_3 = nn.Sequential(        
#                 nn.Conv2d(in_channels=128,
#                           out_channels=256,
#                           kernel_size=(3, 3),
#                           stride=(1, 1),
#                           padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(in_channels=256,
#                           out_channels=256,
#                           kernel_size=(3, 3),
#                           stride=(1, 1),
#                           padding=1),
#                 nn.ReLU(),        
#                 nn.Conv2d(in_channels=256,
#                           out_channels=256,
#                           kernel_size=(3, 3),
#                           stride=(1, 1),
#                           padding=1),
#                 nn.ReLU(),
#                 nn.MaxPool2d(kernel_size=(2, 2),
#                              stride=(2, 2))
#         )
          
#         self.block_4 = nn.Sequential(   
#                 nn.Conv2d(in_channels=256,
#                           out_channels=512,
#                           kernel_size=(3, 3),
#                           stride=(1, 1),
#                           padding=1),
#                 nn.ReLU(),        
#                 nn.Conv2d(in_channels=512,
#                           out_channels=512,
#                           kernel_size=(3, 3),
#                           stride=(1, 1),
#                           padding=1),
#                 nn.ReLU(),        
#                 nn.Conv2d(in_channels=512,
#                           out_channels=512,
#                           kernel_size=(3, 3),
#                           stride=(1, 1),
#                           padding=1),
#                 nn.ReLU(),            
#                 nn.MaxPool2d(kernel_size=(2, 2),
#                              stride=(2, 2))
#         )
        
#         self.block_5 = nn.Sequential(
#                 nn.Conv2d(in_channels=512,
#                           out_channels=512,
#                           kernel_size=(3, 3),
#                           stride=(1, 1),
#                           padding=1),
#                 nn.ReLU(),            
#                 nn.Conv2d(in_channels=512,
#                           out_channels=512,
#                           kernel_size=(3, 3),
#                           stride=(1, 1),
#                           padding=1),
#                 nn.ReLU(),            
#                 nn.Conv2d(in_channels=512,
#                           out_channels=512,
#                           kernel_size=(3, 3),
#                           stride=(1, 1),
#                           padding=1),
#                 nn.ReLU(),    
#                 nn.MaxPool2d(kernel_size=(2, 2),
#                              stride=(2, 2))             
#         )
        
#         self.features = nn.Sequential(
#             self.block_1, self.block_2, 
#             self.block_3, self.block_4, 
#             self.block_5
#         )
            
#         self.classifier = nn.Sequential(
#             nn.Linear(512*4*4, 4096),
#             nn.ReLU(True),
#             nn.Dropout(p=0.5),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(p=0.5),
#             nn.Linear(4096, num_classes),
#         )
             
#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
#         for m in self.modules():
#             if isinstance(m, torch.nn.Conv2d):
#                 # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 # m.weight.data.normal_(0, np.sqrt(2. / n))
#                 m.weight.detach().normal_(0, 0.05)
#                 if m.bias is not None:
#                     m.bias.detach().zero_()
#             elif isinstance(m, torch.nn.Linear):
#                 m.weight.detach().normal_(0, 0.05)
#                 m.bias.detach().detach().zero_()
        
#     def forward(self, x):

#         x = self.features(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         logits = self.classifier(x)

#         return logits
    
# class vgg16(pl.LightningModule):
#     def __init__(self, model, learning_rate):
#         super().__init__()

#         self.learning_rate = learning_rate
#         # The inherited PyTorch module
#         self.model = model

#         # Save settings and hyperparameters to the log directory
#         # but skip the model parameters
#         self.save_hyperparameters(ignore=['model'])

#         # Set up attributes for computing the accuracy
#         self.train_acc = torchmetrics.classification.BinaryAccuracy()
#         self.valid_acc = torchmetrics.classification.BinaryAccuracy()
#         self.test_acc = torchmetrics.classification.BinaryAccuracy(threshold=0.44)
        
#     # Defining the forward method is only necessary 
#     # if you want to use a Trainer's .predict() method (optional)
#     def forward(self, x):
#         return self.model(x)
        
#     # A common forward step to compute the loss and labels
#     # this is used for training, validation, and testing below
#     def _shared_step(self, batch):
#         features, true_labels = batch
#         logits = self(features)
#         loss = torch.nn.functional.cross_entropy(logits, true_labels)
#         predicted_labels = torch.argmax(logits, dim=1)

#         return loss, true_labels, predicted_labels

#     def training_step(self, batch, batch_idx):
#         loss, true_labels, predicted_labels = self._shared_step(batch)
#         self.log("train_loss", loss)
        
#         # To account for Dropout behavior during evaluation
#         self.model.eval()
#         with torch.no_grad():
#             _, true_labels, predicted_labels = self._shared_step(batch)
#         self.train_acc.update(predicted_labels, true_labels)
#         self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)
#         self.model.train()
#         return loss  # this is passed to the optimzer for training

#     def validation_step(self, batch, batch_idx):
#         loss, true_labels, predicted_labels = self._shared_step(batch)
#         self.log("valid_loss", loss)
#         self.valid_acc(predicted_labels, true_labels)
#         self.log("valid_acc", self.valid_acc,
#                  on_epoch=True, on_step=False, prog_bar=True)

#     def test_step(self, batch, batch_idx):
#         loss, true_labels, predicted_labels = self._shared_step(batch)
#         self.test_acc(predicted_labels, true_labels)
#         self.log("test_acc", self.test_acc, on_epoch=True, on_step=False)

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
#         return optimizer