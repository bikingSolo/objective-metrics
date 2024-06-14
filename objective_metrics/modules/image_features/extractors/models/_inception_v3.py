from torch import nn
from torchvision.models import inception_v3

class InceptionHeadless(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = inception_v3(weights="DEFAULT")
        self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)
