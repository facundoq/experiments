# def convbnrelu(input,output):
#         return nn.Sequential(append(nn.Conv2d(input, output, kernel_size=3, padding=1))
#             ,self.conv_layers.append(nn.ELU())
#             ,self.conv_layers.append(nn.BatchNorm2d(output))
#         )

import torch.nn as nn
import torch.nn.functional as F

class ConvBNRelu(nn.Module):

    def __init__(self,input,output):
        super(ConvBNRelu, self).__init__()
        self.name = "ConvBNRelu"
        self.layers=nn.Sequential(
            nn.Conv2d(input, output, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(output),
            )

    def forward(self,x):
        return self.layers.forward(x)

class VGGLike(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(VGGLike, self).__init__()
        self.name = "vgglike"
        h, w, channels = input_shape
        self.conv_layers = nn.Sequential(
            ConvBNRelu(channels, 64),
            ConvBNRelu(64, 64),
            nn.MaxPool2d(2,2),
            ConvBNRelu(64, 128),
            ConvBNRelu(128, 128),
            nn.MaxPool2d(2, 2),
            ConvBNRelu(128, 256),
            ConvBNRelu(256, 256),
            nn.MaxPool2d(2, 2),
            ConvBNRelu(256, 512),
            ConvBNRelu(512, 512),
            nn.MaxPool2d(2, 2),
            Flatten(),
        )
        hf, wf = h // (2 ** 4), w // (2 ** 4)
        flattened_output_size = hf * wf * 512

        self.dense_layers = nn.Sequential(
            nn.Linear(flattened_output_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,num_classes)

        )

    def forward(self, x):

        x=self.conv_layers.forward(x)
        x=self.dense_layers.forward(x)
        x=F.log_softmax(x, dim=1)
        return x




