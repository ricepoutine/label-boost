import os
import json
import torch
from torch import nn, tensor
import sys
from utils import process_state_dict
sys.path.append('.')
from lib.utils.clustering import KNN
from transformers import ViTMAEForPreTraining
from torchmetrics.classification import BinaryAUROC
# trying layernorm instead of:
'''
def softmax(x):
    f = torch.exp(x - torch.max(x))
    return f / f.sum(axis=0) # need to check this!
'''
def softmax(x):
    pass

class CustomLinearClassifier(nn.Sequential):
    def __init__(self, num_classes, in_features=1024):
        super().__init__()
        self.add_module("layernorm", nn.LayerNorm([in_features])) # need to deal with numerical instability
        self.add_module("linear", nn.Linear(in_features=in_features, out_features=num_classes, bias=True))

class CustomNonLinearClassifier(nn.Sequential):
    def __init__(self, num_classes, in_features=[50, 1024]):
        super().__init__()
        input_dim = int(in_features[0]*in_features[1])
        self.add_module("layernorm0", nn.LayerNorm([in_features[1]])) # normalize along the 50 heads dimension
        self.add_module("flatten", nn.Flatten()) # 51200
        self.add_module("hidden1", nn.Linear(in_features=input_dim, out_features=input_dim//8)) #6400
        self.add_module("act1", nn.ReLU())
        self.add_module("layernorm1", nn.LayerNorm([input_dim//8]))
        self.add_module("hidden2", nn.Linear(in_features=input_dim//8, out_features=input_dim//8)) 
        self.add_module("act2", nn.ReLU())
        self.add_module("layernorm2", nn.LayerNorm([input_dim//8]))
        self.add_module("last_hidden", nn.Linear(in_features=input_dim//8, out_features=num_classes)) # 2

class CustomViTMAE(nn.Module):
    def __init__(self, num_classes, vitmae_checkpoint_path, linear=True, classifier_checkpoint_path=None, device=torch.device(device='cuda')):
        super().__init__()
        self.device = device # double check map_location is not keeping them in cpu forever...
        self.linear = linear
        vitmae = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-large")
        pretrained_weights = torch.load(vitmae_checkpoint_path, map_location=device) # must be pretrained
        vitmae.load_state_dict(pretrained_weights['state_dict'])
        self.features = vitmae
        self.features = self.features.to(device=device)
        self.features.eval()
        if linear: self.classifier = CustomLinearClassifier(num_classes=num_classes, in_features=1024).to(device=device)
        else: self.classifier = CustomNonLinearClassifier(num_classes=num_classes, in_features=(50,1024)).to(device)
        if classifier_checkpoint_path is not None: # load pretrained linear weights
            self.classifier.load_state_dict(torch.load(classifier_checkpoint_path, map_location=device)['state_dict'])
            self.classifier = self.classifier.to(device=device)
    
    def forward(self, x):
        # feature extraction (don't need middle layers, so no need to define them)
        # NOTE: make sure gradients are only for classifier. gradient tensor should just be nn.Linear in shape
        with torch.no_grad():
            outputs = self.features(x, output_hidden_states=True)
            if self.linear: feats = torch.sum(outputs.hidden_states[-1], dim=1) # NOTE: could potentially not sum it and use the 50x1024. flatten does exist for a reason
            else: feats = outputs.hidden_states[-1]
        x = self.classifier(feats)
        return torch.softmax(x, dim=1), feats #NOTE: check this, make sure it doesn't result in infinite loss lol

def get_knn_pseudolabels(logger, model, knn, dloader, ar, knna, knn_k, pl_path, pl_type='', num_classes=2, device=torch.device(device='cuda')):
    ### TODO: NEEDS STRICT TESTING!!! MAKE SURE PSEUDOLABELS ARE CORRECTLY INDEXED 
    # evaluate pslabel accuracy on the spot too

    scores = []
    labels = []
    preds = []
    targets = []
    metric = BinaryAUROC()
    print("making pseudolabel set for", str(pl_type))
    pseudolabels = {}
    with torch.no_grad():
        for iter, (image, label, index) in enumerate(dloader):
            image = image.float().to(device)
            label = label.float().to(device)
            label_onehot = torch.zeros(label.size(0), num_classes).to(device).scatter_(1, label.to(torch.int64).view(-1, 1), 1)
            outputs = model(image, output_hidden_states=True)
            features = torch.sum(outputs.hidden_states[-1], dim=1)
            soft_label = torch.Tensor([get_soft_label(knn, features, knn_k)]).to(device)
            pseudolabels[str(index.item())] = soft_label.item() # keep in mind 0.5 gets turned to 0, not 1!
            pred_onehot = torch.zeros(label.size(0), num_classes).to(device).scatter_(1, soft_label.to(torch.int64).view(-1, 1), 1)
            preds.append(soft_label.item())
            targets.append(label.item())
            scores.append(pred_onehot)
            labels.append(label_onehot) 
            #if iter % 10 == 0:
            #    print(f'Pred: {soft_label.item()}, Real label: {label.item()}') # Soft label: {soft_label.item()}, 

    # save pseudolabels in path, ASSUMES ALL SEEDS=0 AND OTHER DEFAULTS (tr=0.7, vr=0.1)
    with open(pl_path+f'/knna{knna}/{pl_type}.json', 'w+') as f:
        json.dump(pseudolabels, f)
    print("pseudolabel set creation complete. Can be found at", pl_path)

    auc = metric(tensor(preds), tensor(targets))
    scores = torch.cat(scores, dim=0).detach()
    labels = torch.cat(labels, dim=0).detach()
    _, y_true = torch.topk(labels, k=1, dim=1)
    _, y_pred = torch.topk(scores, k=num_classes, dim=1)
    top_1_accuracy = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)

    logger.add_scalar(tag=f"Eval/{pl_type} Top 1 accuracy", scalar_value=top_1_accuracy, global_step=int(ar*10))
    logger.add_scalar(tag=f"Eval/{pl_type} AUROC", scalar_value=auc, global_step=int(ar*10))
    print(f"Eval {pl_type} top-1 accuracy:", f"{top_1_accuracy:.5f}")
    print(f"Eval {pl_type} AUROC:", f"{auc:.5f}")

def get_feature_population_and_labels(model, dloader, device=torch.device(device='cuda')):
    # reset feat and label list for umap, embedding visualization on chosen set (train, test).
    feature_list = []
    label_list = []
    with torch.no_grad():
        for image, label, index in dloader:
            image = image.float().to(device)
            outputs = model(image, output_hidden_states=True)
            features = torch.sum(outputs.hidden_states[-1], dim=1)
            if label.size(0) > 1:
                label_list.extend(label.tolist())
                feature_list.extend(features)
            else:
                label_list.append(label.item())
                feature_list.append(features)

        # return these for umap, as we wish to see how well it separates all training images
        population = torch.stack(feature_list, dim=0).view(-1, 1024)
        labels = torch.Tensor(label_list).to(dtype=torch.int64)

    return population, labels


def get_knn(model, dloader_l, k, device=torch.device(device='cuda')):
    feature_list = []
    label_list = []
    model.eval()
    print(f"generating knn population and pseudolabels using k={str(k)}")

    with torch.no_grad():
        for i, (image, label, index) in enumerate(dloader_l):
            image = image.float().to(device)
            outputs = model(image, output_hidden_states=True)
            if label.size(0) > 1: 
                label_list.extend(label.tolist())
                features = torch.sum(outputs.hidden_states[-1], dim=1) # as recommended in https://medium.com/gumgum-tech/image-embeddings-clustered-5cf6a272ac3e
                feature_list.extend(features)
            else: 
                label_list.append(label.item())
                features = torch.sum(outputs.hidden_states[-1], dim=1)
                feature_list.append(features)

    # intermediary: use labeled population to generate pseudolabels
    population = torch.stack(feature_list, dim=0).view(-1, 1024)
    labels = torch.Tensor(label_list).to(dtype=torch.int64)
    knn = KNN(population, labels, k=k)
    return knn

# by = 'vote', 'dist', etc... more disciplined approach tbd
def get_soft_label(knn, features, k=100, by='vote'): 
    with torch.no_grad():
        pred_label, votes, dist = knn(features)
        if by=='vote':
            soft_label = torch.sum(votes)/k
            return soft_label.item()


if __name__ == "__main__":
    device = torch.device(device="cuda")
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    vitmae_chk = './output/SIIM_train_images/ViTMAE_expanded/tr0.7_s0_k0/checkpoints/150.pth.tar'
    model = CustomViTMAE(2, linear=False, vitmae_checkpoint_path=vitmae_chk, device=device)
    with torch.no_grad():
        model.eval()
        print(model.classifier)
        sample = torch.randn([16,3,224,224]).to(device=device)
        prob, feats = model(sample)
        print(prob, feats.shape)

