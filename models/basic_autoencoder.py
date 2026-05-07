### basic autoencoder experiment (practice with pytorch feature extraction and other implementations)
import gin
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary


@gin.configurable
class AutoEncoder(nn.Module):
    def __init__(self, input_size, enc_hidden_dims, z_dim, dec_hidden_dims, num_classes, data_parallel=True, drop_rate=0.0):
        super(AutoEncoder, self).__init__()
        #encoder, hidden_dims[-1] is 'z' vector
        modules = []
        modules.append(nn.Linear(input_size[0]*input_size[1], enc_hidden_dims[0]))
        modules.append(nn.ReLU())
        for i in range(len(enc_hidden_dims)):
            if i < len(enc_hidden_dims)-1:
                modules.append(nn.Linear(enc_hidden_dims[i], enc_hidden_dims[i+1]))
                modules.append(nn.ReLU())
            else:
                modules.append(nn.Linear(enc_hidden_dims[-1], z_dim))

        self.encoder = nn.Sequential(*modules)

        modules = []
        modules.append(nn.ReLU())
        modules.append(nn.Linear(z_dim, dec_hidden_dims[0]))
        modules.append(nn.ReLU())
        for i in range(len(dec_hidden_dims)):
            if i < len(dec_hidden_dims)-1:
                modules.append(nn.Linear(dec_hidden_dims[i], dec_hidden_dims[i+1]))
                modules.append(nn.ReLU())
            else:
                modules.append(nn.Linear(dec_hidden_dims[-1], input_size[0]*input_size[1]))
        self.decoder = nn.Sequential(*modules)

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(z_dim, num_classes)
            )

        if data_parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)
            #self.classifier = nn.DataParallel(self.classifier)

    def forward(self, imgs):
        batch_size = imgs.size(0)
        imgs_flat = imgs.flatten(start_dim=2)
        features = self.encoder(imgs_flat)
        reconstruction = self.decoder(features)
        reconstruction = reconstruction.view(batch_size, imgs.size(1), imgs.size(2), imgs.size(3))
        #result = self.classifier(features)
        return reconstruction, features#, result

if __name__ == "__main__":

    def MSE(input, output):
        return sum((input - output)**2) / len(input)
    
    device="cuda"

    enc_hidden_dims = [256, 128, 64]
    z_dim = 10
    dec_hidden_dims = [64, 128, 256]
    model = AutoEncoder((768,768), enc_hidden_dims, z_dim, dec_hidden_dims, 2, False, None).to(device)
    a = torch.randn(10, 1, 768, 768, requires_grad=True).to(device)
    #print(model)
    reconstruction, features = model(a)
    #print(reconstruction.shape)
    
    loss = F.mse_loss(a, reconstruction).to(device)
    #print(a[0], reconstruction[0])
    #print(loss)
    loss.backward(retain_graph=True)
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(name)
            print(param.data.cpu().numpy().shape)
            print('gradient is \t', param.grad, '\t requires grad:', param.requires_grad, "\n")

