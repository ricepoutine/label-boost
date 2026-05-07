import torch
import torch.nn as nn
from decoder import Decoder
import gin

class _Inference(nn.Sequential):
    def __init__(self, num_input_channels, latent_dim, disc_variable=True):
        super().__init__()
        self.add_module('fc', nn.Linear(num_input_channels, latent_dim))
        if disc_variable:
            self.add_module('log_softmax', nn.LogSoftmax(dim=1))

class Sample(nn.Module):
    def __init__(self, temperature, device="cuda:0"):
        super().__init__()
        self._temperature = temperature
        self.device=device
    
    def forward(self, norm_mean, norm_log_sigma, disc_log_alpha=None, disc_label=None):
        # mixup TBD...
        """
        :param norm_mean: mean vector z_mu
        :param norm_log_sigma: log sigma vector z_s
        :param disc_label: ground truth label (not one-hot encoded)
        ...
        """
        batch_size = norm_mean.size(0)
        latent_sample = list([])
        latent_sample.append(self._sample_norm(norm_mean, norm_log_sigma, device=self.device))

        if disc_log_alpha is not None:
            if disc_label is not None:
                # we have real label, thus no sampling
                # c: N*c onehot
                # mixup not implemented
                c = torch.zeros(disc_log_alpha.size()).to(self.device)
                c = c.scatter(1, disc_label.to(torch.int64).view(-1,1), 1)
                latent_sample.append(c)
            else:
                latent_sample.append(self._sample_gumbel_softmax(disc_log_alpha, self._temperature, device=self.device))
                
        latent_sample = torch.cat(latent_sample, dim=1)
        dim_size = latent_sample.size(1)
        latent_sample = latent_sample.view(batch_size, dim_size, 1, 1)
        return latent_sample
    
    @staticmethod
    def _sample_norm(mu, log_sigma, device="cuda:0"):
        # reparameterization trick for sampling latent vector z from z_mu, z_s
        """
        :param mu: mu for sampling (N*D)
        :param log_sigma: log_sigma for sampling (N*D)
        """
        std_z = torch.randn(mu.size()).to(device) # do we need to check mu.is_cuda()? threw an error saying mu is bool type...
        
        return mu + torch.exp(log_sigma) * std_z

    @staticmethod
    def _sample_gumbel_softmax(log_alpha, temperature, device="cuda:0"):
        """
        Samples from a gumbel-softmax distribution using the reparameterization trick.
        
        :param log_alpha: torch.Tensor, parameters of gumbel-softmax distribution (N*D)
        """
        EPS = 1e-12
        unif = torch.rand(log_alpha.size()).to(device)
        gumbel = -torch.log(-torch.log(unif+EPS) + EPS)
        # Reparameterize to create gumbel softmax sample
        logit = (log_alpha + gumbel) / temperature
        return torch.softmax(logit, dim=1)


###NOTE: dropout yet to be implemented, if needed
@gin.configurable
class M2ResNetVAE(nn.Module): # NOT ACTUALLY M2, its just the light version of label-aware VAE (SHOT-VAE)
    def __init__(self, num_input_channels, image_size, block, n_blocks, channels, resize_factor, cont_latent_dim=None, disc_latent_dim=None, decoder_dim=64, sample_temperature=0.67, data_parallel=True, drop_rate=0.0, device="cuda:0"):
        super().__init__()
        channels = channels
        self.in_channels = channels[0]
        self.device=device
        self.cont_latent_dim = cont_latent_dim
        self.disc_latent_dim = disc_latent_dim
        assert len(n_blocks) == len(channels) # == 4

        pre_process = _PreProcess(num_input_channels, self.in_channels, original=False)
        if data_parallel:
            pre_process = nn.DataParallel(pre_process)

        self.encoder = nn.Sequential()
        self.encoder.add_module('pre_process', pre_process)

        for i, channel in enumerate(channels):
            if i==0:
                resblock = self.get_resnet_layer(block, n_blocks[i], channel)
            else:
                resblock = self.get_resnet_layer(block, n_blocks[i], channel, stride=2)
            if data_parallel:
                resblock = nn.DataParallel(resblock)
            self.encoder.add_module(f"resblock{i+1}", resblock)

        # end of encoder
        if block == Bottleneck:
            trans = _Transition2D(channels[-1]*4)
            self.num_feature_channels = channels[-1]*4
        elif block == BasicBlock:
            trans = _Transition2D(channels[-1])
            self.num_feature_channels = channels[-1]
        else:
            raise NotImplementedError(f"{block} is not implemented")
        if data_parallel:
            trans = nn.DataParallel(trans)
        self.encoder.add_module('transition', trans)

        # turn to 1d latent dim:
        global_avg = nn.AdaptiveAvgPool2d(output_size=(1,1))
        if data_parallel:
            global_avg = nn.DataParallel(global_avg)
        self.global_avg = global_avg
        
        # inference: feature dim to latent sample z; label prediction: feature dim to prediction y
        self.cont_inf = nn.Sequential()
        self.disc_inf = nn.Sequential()
        cont_inf_mean = _Inference(num_input_channels=self.num_feature_channels, latent_dim=cont_latent_dim, disc_variable=False)
        cont_inf_logsigma = _Inference(num_input_channels=self.num_feature_channels, latent_dim=cont_latent_dim, disc_variable=False)
        disc_inf = _Inference(num_input_channels=self.num_feature_channels, latent_dim=disc_latent_dim, disc_variable=True)
        if data_parallel:
            cont_inf_mean = nn.DataParallel(cont_inf_mean)
            cont_inf_logsigma = nn.DataParallel(cont_inf_logsigma)
            disc_inf = nn.DataParallel(disc_inf)
        self.cont_inf.add_module('mean', cont_inf_mean)
        self.cont_inf.add_module('log_sigma', cont_inf_logsigma)
        self.disc_inf = disc_inf # not as module? why?

        # latent sample generator (Decoder)
        sample = Sample(temperature=sample_temperature, device=self.device)
        decoder_kernel_size = tuple([int((s*resize_factor)/32) for s in image_size])
        # if label=latent, then latent_dim=int(continuous_latent_dim + np.sum(disc_latent_dim))
        decoder = Decoder(num_channels=num_input_channels, kernel_size=decoder_kernel_size, num_features=int(decoder_dim/16), latent_dim=cont_latent_dim+disc_latent_dim)
        if data_parallel:
            sample = nn.DataParallel(sample)
            decoder = nn.DataParallel(decoder)
        self.sample = sample
        self.decoder = decoder
    
    def forward(self, input_img, disc_label=None):
        batch_size = input_img.size(0)
        features = self.encoder(input_img)
        avg_features = self.global_avg(features).view(batch_size, -1)
        norm_mean = self.cont_inf.mean(avg_features)
        norm_log_sigma = self.cont_inf.log_sigma(avg_features)
        disc_log_alpha = self.disc_inf(avg_features)
        latent_sample = self.sample(norm_mean=norm_mean, norm_log_sigma=norm_log_sigma, disc_log_alpha=disc_log_alpha, disc_label=disc_label)
        #if self.eval():
        #    return latent_sample, disc_log_alpha
        #else: 
        reconstruction = self.decoder(latent_sample)
        return reconstruction, norm_mean, norm_log_sigma, disc_log_alpha
        
    def get_resnet_layer(self, block, n_blocks, channels, stride=1):
        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False

        layers = []
        layers.append(block(self.in_channels, channels, stride, downsample))
        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels

        return nn.Sequential(*layers)
        
    def get_resnet_layer(self, block, n_blocks, channels, stride=1):
        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False

        layers = []
        layers.append(block(self.in_channels, channels, stride, downsample))
        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels

        return nn.Sequential(*layers)
    
###NOTE: dropout yet to be implemented, if needed
@gin.configurable
class M1ResNetVAE(nn.Module):
    # same as regular ResNetVAE, but without label inference. 
    # This is for a detached system which first uses M1VAE to learn z,
    # then gives z to a small MLP model to predict y. Losses and gradients are separate!
    def __init__(self, num_input_channels, image_size, block, n_blocks, channels, resize_factor, cont_latent_dim=None, decoder_dim=64, sample_temperature=None, data_parallel=True, drop_rate=0.0, device="cuda:0"):
        super().__init__()
        channels = channels
        self.in_channels = channels[0]
        self.device=device
        self.cont_latent_dim = cont_latent_dim
        assert len(n_blocks) == len(channels) # == 4

        pre_process = _PreProcess(num_input_channels, self.in_channels, original=False)
        if data_parallel:
            pre_process = nn.DataParallel(pre_process)

        self.encoder = nn.Sequential()
        self.encoder.add_module('pre_process', pre_process)

        for i, channel in enumerate(channels):
            if i==0:
                resblock = self.get_resnet_layer(block, n_blocks[i], channel)
            else:
                resblock = self.get_resnet_layer(block, n_blocks[i], channel, stride=2)
            if data_parallel:
                resblock = nn.DataParallel(resblock)
            self.encoder.add_module(f"resblock{i+1}", resblock)

        # end of encoder
        if block == Bottleneck:
            trans = _Transition2D(channels[-1]*4)
            self.num_feature_channels = channels[-1]*4
        elif block == BasicBlock:
            trans = _Transition2D(channels[-1])
            self.num_feature_channels = channels[-1]
        else:
            raise NotImplementedError(f"{block} is not implemented")
        if data_parallel:
            trans = nn.DataParallel(trans)
        self.encoder.add_module('transition', trans)

        # turn to 1d latent dim:
        global_avg = nn.AdaptiveAvgPool2d(output_size=(1,1))
        if data_parallel:
            global_avg = nn.DataParallel(global_avg)
        self.global_avg = global_avg
        
        # inference: feature dim to latent dim z
        self.cont_inf = nn.Sequential()
        cont_inf_mean = _Inference(num_input_channels=self.num_feature_channels, latent_dim=cont_latent_dim, disc_variable=False)
        cont_inf_logsigma = _Inference(num_input_channels=self.num_feature_channels, latent_dim=cont_latent_dim, disc_variable=False)
        if data_parallel:
            cont_inf_mean = nn.DataParallel(cont_inf_mean)
            cont_inf_logsigma = nn.DataParallel(cont_inf_logsigma)
        self.cont_inf.add_module('mean', cont_inf_mean)
        self.cont_inf.add_module('log_sigma', cont_inf_logsigma)

        # latent sample generator (Decoder)
        sample = Sample(temperature=sample_temperature, device=self.device)
        decoder_kernel_size = tuple([int((s*resize_factor)/32) for s in image_size])
        # if label=latent, then latent_dim=int(continuous_latent_dim + np.sum(disc_latent_dim))
        decoder = Decoder(num_channels=num_input_channels, kernel_size=decoder_kernel_size, num_features=decoder_dim, latent_dim=cont_latent_dim)
        if data_parallel:
            sample = nn.DataParallel(sample)
            decoder = nn.DataParallel(decoder)
        self.sample = sample
        self.decoder = decoder
    
    def forward(self, input_img):
        batch_size = input_img.size(0)
        features = self.encoder(input_img)
        avg_features = self.global_avg(features).view(batch_size, -1)
        norm_mean = self.cont_inf.mean(avg_features)
        norm_log_sigma = self.cont_inf.log_sigma(avg_features)
        latent_sample = self.sample(norm_mean=norm_mean, norm_log_sigma=norm_log_sigma)
        #if self.eval: ###### DISABLE TO TRAIN M1! THIS EVAL LOGIC IS FOR AFTER PRETRAINING!
        #    return norm_mean, norm_log_sigma
        #else:
        reconstruction = self.decoder(latent_sample)
        return reconstruction, norm_mean, norm_log_sigma, latent_sample
        
    def get_resnet_layer(self, block, n_blocks, channels, stride=1):
        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False

        layers = []
        layers.append(block(self.in_channels, channels, stride, downsample))
        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels

        return nn.Sequential(*layers)

class _PreProcess(nn.Sequential):
    def __init__(self, num_input_channels, num_init_features, original=False):
        super().__init__()
        """
        if small_input:
            self.add_module('conv0', nn.Conv2d(num_input_channels, num_init_features, kernel_size=3, stride=1, padding=1, bias=True))
        """
        if original:
            self.add_module('conv0', nn.Conv2d(num_input_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=True))
            self.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))
        else: #using Ross tutorial
            self.add_module('conv0', nn.Conv2d(num_input_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False))
            self.add_module('bn0', nn.BatchNorm2d(num_init_features))
            self.add_module('relu0', nn.ReLU(inplace=True))
            self.add_module('maxpool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) #optional?
    
    """
    def forward(self, x): #slower, but needed for debug
    for module in self:
        x = module(x)
        print(module, x.shape)
    return x # turns out this reduces image size by factor of 4... [ex: 768 -> 384 -> 192]
    """


class _Transition2D(nn.Sequential):
    def __init__(self, num_input_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.LeakyReLU(inplace=True))
        #self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        #self.add_module('pool', nn.AvgPool2d(kernel_size=1, stride=1, padding=1))

@gin.configurable
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        
        if downsample:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False)
            bn = nn.BatchNorm2d(out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None
        
        self.downsample = downsample
        
    def forward(self, x):
        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.downsample is not None:
            i = self.downsample(i)

        x += i
        x = self.relu(x)

        return x

@gin.configurable
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
        super().__init__()
    
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size = 1, stride = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)
        
        self.relu = nn.ReLU(inplace = True)
        
        if downsample:
            conv = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size = 1, 
                             stride = stride, bias = False)
            bn = nn.BatchNorm2d(self.expansion * out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None
            
        self.downsample = downsample
        
    def forward(self, x):
        i = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.downsample is not None:
            i = self.downsample(i)
        
        x += i
        x = self.relu(x)
    
        return x

def evaluation(model, dloader, device='cuda'):
    feature_list = []
    label_list = []
    model.eval()
    with torch.no_grad():
        for image, label in dloader:
            image = image.float().to(device)
            recon, mu, sigma, latent = model(image)
            label_list.append(label.item())
            feature = latent.detach().cpu()
            feature = feature.view((1,64))
            feature_list.append(feature) # random feature slice

    population = torch.stack(feature_list, dim=0).view(-1, 64)
    
    #get_tsne(population, label_list, n=2)
    get_pca(population, label_list, n=2)

if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from lib.evaluation import get_pca, get_tsne
    from lib.dataloader import create_dataloaders
    import numpy as np
    from pathlib import Path
    
    tr, ar = (0.8,0.6)
    device = torch.device(device='cuda')
    checkpoint = 'checkpoint_100'

    model_seed = 0
    model_name = 'm1_b0'

    data_seed = 0
    torch.manual_seed(data_seed)
    torch.cuda.manual_seed(data_seed)
    np.random.seed(data_seed)

    train_dloader, test_dloader = create_dataloaders(dataset='hip734_cropped', train_ratio=tr, \
                    ssl=False, expanded_test_set=False, annot_ratio=ar, num_workers=4, num_output_channels=1, image_size=(224,224))
    
    model = M1ResNetVAE(num_input_channels=1, image_size=(128,128), block=Bottleneck, n_blocks=[3,4,6,3], channels=[64, 128, 256, 512], resize_factor=1.0, cont_latent_dim=64, decoder_dim=128, sample_temperature=None, data_parallel=True)
    model_path = Path.cwd() / 'output' / 'M1ResNetVAE' / 'hip734_cropped' / f'tr{tr}_s{data_seed}_{model_name}_s{model_seed}' / 'checkpoints' / f'{checkpoint}.pth.tar'
    model_weights = torch.load(model_path)
    model.load_state_dict(model_weights['state_dict'])
    model = model.to(device)

    evaluation(model, train_dloader)

"""
# defunct code, awaiting deletion at the end of this project
    import gin
    import argparse
    import sys
    sys.path.append(".")
    from lib.dataloader import lc20000_dataset, bc780_dataset
    from lib.criterion import VAECriterion
    from torch.utils.data import DataLoader
    from lib.utils.plot import plot_images
    from pathlib import Path
    device = "cuda"
    device = torch.device(device)
    dpath = Path.cwd() / 'datasets' / 'lc20000'

    #ResNetConfig = namedtuple('ResNetConfig', ['color_channels', 'block', 'n_blocks', 'channels'])
    #resnet50_config = ResNetConfig(color_channels=1, 
    #                               block = Bottleneck, 
    #                               n_blocks = [3, 4, 6, 3], 
    #                               channels = [512, 256, 128, 64]
    #                               )

    train_dataset, _ = bc780_dataset(dpath)
    num_classes = 3
    batch_size = 1
    color_channels = 1
    img_size = (224, 224)
    latent_dim = 1024
    beta_kl = 0.1
    dloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    model = M2ResNetVAE(num_input_channels=color_channels, sample_temperature=0.67, block=Bottleneck, n_blocks=[3, 4, 6, 3], channels=[512, 256, 128, 64], image_size=img_size, resize_factor=1.0, cont_latent_dim=latent_dim, disc_latent_dim=num_classes, decoder_dim=512, device='cuda')
    model = model.to(device)
    #print(model)
    # check immediate outputs:
    #a = torch.randn(batch_size, color_channels, img_size[0], img_size[1])
    #model(a)
    model.train()
    elbo_criterion = VAECriterion(discrete_dim=None, x_sigma=10, bce_reconstruction=True).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=5e-4) #wd makes weights approach 0, momentum changes how quickly you reach point
    # train loop debug #NOTE: gradients are still exploding, and recon loss seems to drown out kl loss (unless gradient calcs are different... look into this)
    for epoch in range(3):
        for i, (image, label) in enumerate(dloader):
            image = image.float().to(device)
            label = label.float().to(device)
            reconstruction, norm_mean, norm_log_sigma, latent_sample = model(image, disc_label=None)
            reconstruction_loss, cont_prior_kl_loss, _ = elbo_criterion(image, reconstruction, norm_mean, norm_log_sigma)
            elbo_loss = reconstruction_loss + beta_kl*cont_prior_kl_loss #should this be weighted?
            if i % 100 == 0: 
                print(reconstruction_loss.item(), cont_prior_kl_loss.item())
            elbo_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
    # look at one reconstructed image after one epoch:
    model.eval()
    with torch.no_grad():
        for i, (image, label) in enumerate(dloader):
            if i == 0:
                image = image.float().to(device)
                label = label.float().to(device)
                reconstruction, *_ = model(image, disc_label=None)
                print(torch.sigmoid(reconstruction))
                plot_images(images=image, labels=None, normalize=True, fname='Real_image_test')
                plot_images(images=torch.sigmoid(reconstruction), labels=None, normalize=True, fname='Reconstructed_image_test')
            else:
                break
    
    print('test complete')
"""


        



