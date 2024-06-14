"""
Architecture of SPAQ model.

src: https://github.com/h4nwei/SPAQ
"""
import torch.nn as nn
import torchvision

class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=False)
        fc_feature = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(fc_feature, 1, bias=True)

    def forward(self, x):
        result = self.backbone(x)
        return result

class MTA(nn.Module):
    def __init__(self):
        super(MTA, self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=False)
        fc_feature = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(fc_feature, 6, bias=True)

    def forward(self, x):
        result = self.backbone(x)
        return result
    
class MTS(nn.Module):
    def __init__(self):
        super(MTS, self).__init__()
        # self.config = config
        self.backbone_semantic = torchvision.models.resnet50(pretrained=False)
        self.backbone_quality = torchvision.models.resnet50(pretrained=False)
        fc_feature = self.backbone_quality.fc.in_features
        self.backbone_quality.fc = nn.Linear(fc_feature, 1, bias=True)
        self.backbone_semantic.fc = nn.Linear(fc_feature, 9, bias=True)

    def forward(self, x):
        batch_size = x.size()[0]

        # Shared layers
        x = self.backbone_quality.conv1(x)
        x = self.backbone_quality.bn1(x)
        x = self.backbone_quality.relu(x)
        x = self.backbone_quality.maxpool(x)
        x = self.backbone_quality.layer1(x)
        x = self.backbone_quality.layer2(x)
        x = self.backbone_quality.layer3(x)

        # Image quality task
        x1 = self.backbone_quality.layer4(x)
        x2 = self.backbone_quality.avgpool(x1)
        x2 = x2.squeeze(2).squeeze(2)

        quality_result = self.backbone_quality.fc(x2)
        quality_result = quality_result.view(batch_size, -1)

        # Scen semantic task
        xa = self.backbone_semantic.layer4(x)
        xb = self.backbone_semantic.avgpool(xa)
        xb = xb.squeeze(2).squeeze(2)

        semantic_result = self.backbone_semantic.fc(xb)
        semantic_result = semantic_result.view(batch_size, -1)

        return quality_result, semantic_result