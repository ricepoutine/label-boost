from torch import nn
import torch

# basic decoder, basically unchanged from FengZH
# seems like kernel_size controls how big the reconstruction will be. should scale to the same dims as input

class Decoder(nn.Module):
    def __init__(self, num_channels=1, kernel_size=(5,6), num_features=64, latent_dim=100, data_parallel=True):
        super(Decoder, self).__init__()
        decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=num_features*16, kernel_size=kernel_size, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=num_features*16),
            nn.ReLU(inplace=True),
            # state size: (num_features*32) x 5 x 6
            nn.ConvTranspose2d(in_channels=num_features*16, out_channels=num_features*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=num_features*8),
            nn.ReLU(inplace=True),
            # state size: (num_features*16) x 10 x 12
            nn.ConvTranspose2d(in_channels=num_features*8, out_channels=num_features*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=num_features*4),
            nn.ReLU(inplace=True),
            # state size: (num_features*8) x 20 x 24
            nn.ConvTranspose2d(in_channels=num_features*4, out_channels=num_features*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=num_features*2),
            nn.ReLU(inplace=True),
            # state size: (num_features*4) x 40 x 48
            nn.ConvTranspose2d(in_channels=num_features*2, out_channels=num_features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=num_features),
            nn.ReLU(inplace=True),
            # state size: (num_features*2) x 80 x 96
            nn.ConvTranspose2d(num_features, num_channels, 4,2,1,bias=False),
            # activation functions here?
            # state size w/o sigmoid function: num_channel x 160 x 192
            #nn.Sigmoid() # we do sigmoid prior to calculating reconstruction loss
        )
        if data_parallel:
            self.decoder = nn.DataParallel(decoder)
        else:
            self.decoder = decoder

    def forward(self, latent_sample):
        return self.decoder(latent_sample)

if __name__ == "__main__":
    resize_factor = 0.5
    img_size = 768
    decoder_kernel_size = (int((img_size * resize_factor)/32),int((img_size * resize_factor)/32))
    cont_latent_dim = 16
    disc_latent_dim = 2
    num_features = 128
    latent_dim = int(cont_latent_dim + disc_latent_dim)
    decoder = Decoder(num_channels=1, kernel_size=decoder_kernel_size, num_features=int(num_features/16), latent_dim=latent_dim, data_parallel=True).cuda()

    # conv2d expected [batch_size, channels, height, width]
    a = torch.randn(1, latent_dim, 1, 1).cuda()
    #print(decoder)
    out = decoder(a)
    print(out, out.shape)
