import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from utils import ReverseLayerF
import torch

class DANN(nn.Module):
    def __init__(self, num_classes=2, num_domains=2):
        super(DANN, self).__init__()
        self.feature = models.resnet18(pretrained=True)
        #self.feature.fc = nn.Linear(2048, 1024)
        self.class_classifier = Classifier(num_classes=num_classes)
        self.domain_classifier = Classifier(num_classes=num_domains)

    def forward(self, x, alpha, source=True):
        x = self.feature(x)
        x = x.view(-1, 1000) # Default ResNet18 FC output size
        reverse_feature = ReverseLayerF.apply(x, alpha)
        
        if source:
            class_output = self.class_classifier(x)
        else:
            class_output = None
        domain_output = self.domain_classifier(reverse_feature)
        
        return class_output, domain_output
    
class Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=1000, out_features=1024), # Default ResNet18 FC output size
            nn.ReLU(),
            # nn.Linear(in_features=100, out_features=100),
            # nn.ReLU(),
            nn.Linear(in_features=1024, out_features=num_classes)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x
