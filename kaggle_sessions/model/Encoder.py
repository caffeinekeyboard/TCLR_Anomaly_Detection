import torch.nn as nn
from torchvision.models import resnet50

class BasicEncoder(nn.Module):
    """
        Light convolutional neural network encoder.\n
        33582208 trainable parameters.
    """

    def __init__(self):
        super(BasicEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, 64)


    def forward(self, x):
        x = self.pool1(nn.functional.relu(self.conv1(x)))
        x = self.pool2(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 64 * 64)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

    
class ResNet50Encoder(nn.Module):
  """
        ResNet-18 encoder.\n
        11209344 trainable parameters.
  """

  def __init__(self):
    super(ResNet50Encoder, self).__init__()
    self.resnet = resnet50(pretrained=True)
    self.in_features = self.resnet.fc.in_features
    self.resnet.fc = nn.Identity()
    self.fc = nn.Linear(self.in_features, 64)


  def forward(self, x):
    x = self.resnet(x)
    x = x.view(x.size(0), -1)
    embedding = self.fc(x)
    return embedding