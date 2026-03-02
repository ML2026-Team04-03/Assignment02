import torch
import torch.nn as nn

from torchvision import models
'''
class ResNetMultimodalClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        try:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        except AttributeError:
            resnet = models.resnet50(pretrained=True)

        # self.image_features = nn.Sequential(*list(resnet.children())[:-1])
        modules = list(resnet.children())[:-2]
        self.image_features = nn.Sequential(*modules)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Added
        self.image_fc = nn.Linear(2048, 512)

        self.text_fc = nn.Sequential(
            nn.Linear(vocab_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, text_vec):
        img = self.image_features(images)
        img = self.avgpool(img)
        img = torch.flatten(img, 1)
        img = self.image_fc(img)
        txt = self.text_fc(text_vec)
        fused = torch.cat((img, txt), dim=1)
        return self.classifier(fused)
    
'''

class EfficientNetV2MMultimodalClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, train_backbone=False):
        super().__init__()
        # EfficientNetV2-M backbone (pretrained)
        try:
            effnet = models.efficientnet_v2_m(
                weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1
            )
        except AttributeError:
            effnet = models.efficientnet_v2_m(pretrained=True)

        # Last conv feature extractor
        self.image_features = effnet.features  # outputs [B, C, H, W]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # EfficientNetV2-M final feature dim is 1280
        self.image_fc = nn.Linear(1280, 512)

        # Optionally freeze backbone for transfer learning
        if not train_backbone:
            for p in self.image_features.parameters():
                p.requires_grad = False

        # Text branch (BoW/TF-IDF vector -> MLP)
        self.text_fc = nn.Sequential(
            nn.Linear(vocab_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU()
        )

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, text_vec):
        img = self.image_features(images)   
        img = self.avgpool(img)             
        img = torch.flatten(img, 1)         
        img = self.image_fc(img)            

        txt = self.text_fc(text_vec)        
        fused = torch.cat((img, txt), dim=1)
        return self.classifier(fused)       
