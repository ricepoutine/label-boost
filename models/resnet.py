import torch
from torch import nn
from transformers import ResNetForImageClassification, ResNetConfig
from utils import process_state_dict

class CustomClassifier(nn.Sequential):
    def __init__(self, num_classes):
        super().__init__()
        self.add_module("0", nn.Flatten())
        self.add_module("1", nn.Linear(in_features=512, out_features=num_classes, bias=True))

class CustomResNet(nn.Module):
    def __init__(self, num_channels,num_classes, pretrained=False, device=torch.device(device='cuda')):
        super().__init__()
        self.device=device
        if pretrained:
            pretrained_resnet = ResNetForImageClassification.from_pretrained('microsoft/resnet-18')
            self.features = nn.Sequential(*list(pretrained_resnet.children())[:-1])
            self.features.load_state_dict(process_state_dict(pretrained_resnet.state_dict().items(), 'resnet'))
        else: # non-pretrained will have 1 channel instead of 3...
            config = ResNetConfig(num_channels=num_channels, embedding_size=64, hidden_sizes=[64, 128, 256, 512], depths=[2,2,2,2],\
                                  layer_type='basic', hidden_act='relu', downsample_in_first_stage=False, out_features=['stage4'],\
                                    out_indices=[4], num_labels=num_classes, stage_names=['stem', 'stage1', 'stage2', 'stage3', 'stage4'])
            resnet = ResNetForImageClassification(config)
            self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.features = self.features.to(device=device)
        self.classifier = CustomClassifier(num_classes=num_classes).to(device=device)
    
    def forward(self, x):
        # feature extraction (don't need middle layers, so no need to define them)
        feats = (self.features(x)).pooler_output
        x = self.classifier(feats)
        return torch.softmax(x, dim=1), feats #NOTE: check this, make sure it doesn't result in infinite loss lol

if __name__ == "__main__":
    device = torch.device(device="mps")
    model = CustomResNet(2, True, device=device)
    with torch.no_grad():
        model.eval()
        sample = torch.randn([16,3,224,224]).to(device=device)
        out = model(sample)
        print(out[0].shape)