# CNN used in Roberto's hip dysplasia experiment (sadly depreciated...)
import torch
import torch.nn as nn
from models.vae import M1ResNetVAE, M2ResNetVAE

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(*self.shape)

class Noise(nn.Module): # should be torch.normal(mean, std, size=()) if we want adjustable noise
    def __init__(self, size, device='cuda'):
        super(Noise, self).__init__()
        self.size = size
        self.device = device
    def forward(self, image):
        # ignore image
        noise_vector = torch.randn(self.size).to(self.device)
        return noise_vector


#@gin.configurable
class VegaCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, data_parallel=True, encoder=None, encoder_dim=64, batch_size=1):
        super().__init__()
        # NOTE: with encoder giving 1d latent of size 64... would it make sense to just cast into 2d? 
        # do we need reroute? or just use flatten?
        # naive: 1d->2d, keep convolution layers... (I have a feeling this might not work...)

        self.encoder = nn.Sequential()
        layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )
        self.encoder.add_module('layer1', layer1)
        layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )
        self.encoder.add_module('layer2', layer2)
        layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )
        self.encoder.add_module('layer3', layer3)
        layer4 = nn.Sequential(
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=(2,2)) # removed else image too small to do final layer... NOTE: notify Roberto
        )
        self.encoder.add_module('layer4', layer4)
        layer5 = nn.Sequential(
            nn.Conv2d(256, 512, 3),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(),
        )
        self.encoder.add_module('layer5', layer5)
        self.encoder.add_module('flatten', nn.Flatten())

        self.pt_encoder = None
        if encoder is not None:
            self.int_layer = nn.Sequential(# made smaller as 1000 might be too big...
                nn.Linear(512*4*4, 1000),
                nn.ReLU(),
            )
            if isinstance(encoder, M1ResNetVAE):
                self.out_layer = nn.Sequential(
                    nn.Linear(1000+(encoder_dim*2), 1000+(encoder_dim*2)),
                    nn.ReLU(),
                    #nn.Linear(1000+(encoder_dim*2), 1000+(encoder_dim*2)), # extra staging layer
                    #nn.ReLU(),
                    nn.Linear(1000+(encoder_dim*2), out_channels)

                )   
            elif isinstance(encoder, M2ResNetVAE): # forcing model to learn how to use encoder's label prediction
                self.out_layer = nn.Sequential(
                    nn.Linear(1000+(encoder_dim*2)+encoder.disc_latent_dim, 1000+(encoder_dim*2)+encoder.disc_latent_dim),
                    nn.ReLU(),
                    nn.Linear(1000+(encoder_dim*2)+encoder.disc_latent_dim, 1000+(encoder_dim*2)+encoder.disc_latent_dim),
                    nn.ReLU(),
                    nn.Linear(1000+(encoder_dim*2)+encoder.disc_latent_dim, out_channels),
                )   
                #self.out_layer2 = nn.Sequential(
                #    nn.GELU(),
                #    nn.Linear(out_channels+1, out_channels)
                #    )
                #if data_parallel:
                #    self.out_layer2 = nn.DataParallel(self.out_layer2)
            if encoder != 'noise':
                # extra failsafe to ensure all layers are frozen
                #encoder.eval()
                #for param in encoder.parameters():  
                #    param.requires_grad = False
                self.pt_encoder = nn.Sequential(
                    encoder,
                )    
            else: # noise vector added to prevent overfitting (latent contribution)
                self.pt_encoder = nn.Sequential(
                    Noise(size=encoder_dim),
                    View(shape=(batch_size, encoder_dim))
                )

        else:
            if in_channels==1: #also means image_size=(128,128)
                self.int_layer = nn.Sequential(
                    nn.Linear(512*4*4, 1000), # for image_size=(128,128), need (512*4*4, 1000)
                    nn.ReLU(),
                    nn.Linear(1000, 1000), #NOTE: some tests use an additional layer here. for consistency (and if you aren't sure), rerun CNN_only
                )
            else:
                self.int_layer = nn.Sequential(
                    nn.Linear(512*4*4*4*4, 1000), # for image_size=(128,128), need (512*4*4, 1000)
                    nn.ReLU(),
                    nn.Linear(1000, 1000), #NOTE: some tests use an additional layer here. for consistency (and if you aren't sure), rerun CNN_only
                )
            self.out_layer = nn.Sequential(
                nn.ReLU(),
                nn.Linear(1000, out_channels)
            )

        if data_parallel:
            self.encoder = nn.DataParallel(self.encoder)
            if encoder is not None:
                self.pt_encoder = nn.DataParallel(self.pt_encoder)
            self.int_layer = nn.DataParallel(self.int_layer)
            self.out_layer = nn.DataParallel(self.out_layer)
    
    # for evaluation, return features rather than prob predictions
    def forward(self, image):
        x = self.encoder(image)
        features = self.int_layer(x)
        x = self.out_layer(features)
        x = torch.sigmoid(x)
        x = torch.stack((1-x, x), dim=-1).squeeze(dim=1)
        return x, features

    # depreciated, since this methodology did not work...
    """ 
    def forward(self, image):
        x = self.encoder(image)
        if self.pt_encoder:
            recon, mu, sigma, _ = self.pt_encoder(image)
            mu = mu.view(1,-1)
            sigma = sigma.view(1,-1)
            x = self.int_layer(x)
            x = torch.cat((x,mu,sigma),dim=1)
            x = self.out_layer(x)
            x = torch.sigmoid(x)
            return x, recon, mu, sigma
        x = self.out_layer(x)
        x = torch.sigmoid(x)
        return x
    """

    
    #hard-coded at batch_size = 1, and label prediction used alongside latent
    # NOTE: currently only label-aware latent, m2 label prediction unknown what to do with...
    def forward_m2(self, image):
        x = self.encoder(image)
        x = self.int_layer(x)
        latent, disc_log_alpha = self.pt_encoder(image)
        latent = latent.view(1, -1)
        #print(x.shape, latent.shape)
        #enc_pred = torch.exp(disc_log_alpha)[0][0].view((1,-1))
        x = torch.cat((x, latent), dim=1)
        x = self.out_layer(x)
        x = torch.sigmoid(x)
        # NOTE: this treats tensor([[0.1, 0.9]]) as [0.1] = prediction, in line with CNN predictions
        # NOTE: also the contribution of encoder label predictions must be learned!
        #x = torch.cat((x, enc_pred), dim=1)
        #x = self.out_layer2(x)
        #x = torch.sigmoid(x) # simply to enforce 0-1 range. NOTE: issue with doing this twice?
        return x
        

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(".")
    from lib.dataloader import bc780_dataset, get_dataset, load_datasets
    from lib.criterion import ClsCriterion
    from torch.utils.data import DataLoader

    device = torch.device('cuda:0')
    dpath = dpath = Path.cwd() / 'datasets' / 'hip734_cropped'
    td, vd, ttd = load_datasets('hip734_cropped', 2)

    # defaults from vega experiment
    num_classes = 2
    batch_size = 1
    lr = 1e-5
    model = VegaCNN(in_channels=1, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()
    
    #a = torch.randn((1,1,128,128)).to(device)
    #out = model(a)
    #print(out.item())

    criterion = ClsCriterion()
    for epoch in range(10):
        #train
        model.train()
        for i, (image, label) in enumerate(td):
            
            optimizer.zero_grad()
            batch_size = image.size(0)
            image = image.float().to(device)
            #print(image.shape)
            label = label.float().to(device)
            
            label_onehot = torch.zeros(label.size(0), num_classes).to(device).scatter_(1, label.to(torch.int64).view(-1, 1), 1)
            pred = model(image) # sigmoid activation [0,1], needs log_softmax
            pred_onehot = torch.log(torch.stack((1-pred, pred), dim=-1).squeeze(dim=1))

            loss = criterion(pred_onehot, label_onehot)
            if i % 100 == 0:
                #print(pred.item(), label.item())
                print(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #validate
        model.eval()
        all_score = []
        all_label = []
        for i, (image, label) in enumerate(ttd):
            batch_size = image.size(0)
            image = image.float().to(device)
            #print(image.shape)
            label = label.float().to(device)
            label_onehot = torch.zeros(label.size(0), num_classes).to(device).scatter_(1, label.to(torch.int64).view(-1, 1), 1)
            pred = model(image)
            pred_onehot = torch.stack((1-pred, pred), dim=-1).squeeze(dim=1)
            all_score.append(pred_onehot)
            all_label.append(label_onehot)
            if i % 100 == 0:
                print(pred.item(), label)
        all_score = torch.cat(all_score, dim=0).detach()
        all_label = torch.cat(all_label, dim=0).detach()
        print(all_score)
        print(all_label)
        _, y_true = torch.topk(all_label, k=1, dim=1)
        _, y_pred = torch.topk(all_score, k=num_classes, dim=1)
        # calculate accuracy by hand
        test_top_1_accuracy = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
        print("top-1 accuracy:", f"{test_top_1_accuracy:.5f}")
        # it works!!! but its better than vega
