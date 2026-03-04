import torch
import torch.nn as nn
from torchvision import models


class EfficientNetV2MMultimodalClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, train_backbone=False,
                 text_input_dropout=0.1, text_modality_dropout=0.3):
        super().__init__()

        # EfficientNetV2-M backbone (pretrained)
        try:
            effnet = models.efficientnet_v2_m(
                weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1
            )
        except AttributeError:
            effnet = models.efficientnet_v2_m(pretrained=True)

        # Dropout for text input and modality to reduce overreliance on text features in training
        self.text_input_dropout = nn.Dropout(p=float(text_input_dropout))
        self.text_modality_dropout = float(text_modality_dropout)

        # Last conv feature extractor
        self.image_features = effnet.features  # outputs [B, C, H, W]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        self.image_fc = nn.Linear(1280, 512)

        # Optionally, freeze backbone for transfer learning
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

        # Fusion classifier with label smoothing support
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

        txt_in = self.text_input_dropout(text_vec)

        if self.training and self.text_modality_dropout > 0:
            # Dropping entire text modality per-sample
            mask = (torch.rand(txt_in.size(0), 1, device=txt_in.device) >
                    self.text_modality_dropout).float()
            txt_in = txt_in * mask

        txt = self.text_fc(txt_in)
        fused = torch.cat((img, txt), dim=1)
        return self.classifier(fused)
