import os
import json
import torch
from torch import nn, tensor
import sys
from utils import process_state_dict
sys.path.append('.')
from lib.utils.clustering import KNN
from models.MedImageInsights.medimageinsightmodel import MedImageInsight
from torchmetrics.classification import BinaryAUROC
from pathlib import Path

class CustomClassifier(nn.Sequential):
    def __init__(self, num_classes):
        super().__init__()
        self.add_module("0", nn.Flatten())
        self.add_module("1", nn.Linear(in_features=1024, out_features=num_classes, bias=True))

class CustomMII(nn.Module):
    def __init__(self, num_classes, classifier_checkpoint_path=None, device=torch.device(device='mps')):
        super().__init__()
        self.device = device
        MII_path = Path.cwd() / 'models' / 'MedImageInsights'
        self.features = MedImageInsight(model_dir= MII_path / '2024.09.27',
            vision_model_name= 'medimageinsigt-v1.0.0.pt',
            language_model_name= 'language_model.pt'
        )
        self.features.load_model()
        self.classifier = CustomClassifier(num_classes=num_classes).to(device=self.device)
        if classifier_checkpoint_path is not None:
            self.classifier.load_state_dict(torch.load(classifier_checkpoint_path, map_location=self.device)['state_dict'])
        self.classifier = self.classifier.to(device=self.device)

    def get_image_embeddings(self, image, norm=False):
        return self.model.get_image_embeddings(image, norm=norm)

    def forward(self, image):
        feats = self.features.get_image_embeddings(image, norm=False)
        x = self.classifier(feats)
        return torch.softmax(x, dim=1), feats
    
if __name__ == "__main__":
    device = torch.device(device="cuda")
    model = CustomMII(2, device=device)
    with torch.no_grad():
        sample = torch.randn([16,3,224,224]).to(device=device)
        pred, feats = model(sample)
        print(pred.shape)