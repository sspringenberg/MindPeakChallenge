from torch import nn, optim
from torchvision import models
import sys

class TransferCNN(nn.Module):
    """
    costruct CNN (transfer learning from VGG11) used for classification
    """
    def __init__(self):
        super(TransferCNN, self).__init__()

        self.net = models.vgg11_bn(pretrained=True)
        self.net.features = self.net.features[0:21]
        for param in self.net.parameters():
            param.requires_grad = False

        self.net.classifier = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size = 1, stride = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, kernel_size = 1, stride = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(64, 4, kernel_size = 1, stride = 1),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        out = self.net.features(input)
        out = self.net.classifier(out)
        out = out.squeeze()

        return out


class CNN(nn.Module):
    """
    costruct CNN (trained from scratch) used for classification
    """

    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size = 3, stride = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.3),

            nn.Conv2d(16, 16, kernel_size = 3, stride = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.3),

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(16, 4, kernel_size = 1, stride = 1),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        out = self.net(input)
        out = out.squeeze()

        return out
