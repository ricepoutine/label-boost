import torch
from torch import nn
import torch.nn.functional as F

eps = 1e-7

###NOTE: not fully understood. please remember to refresh how these criterion work, and thus
# how to augment them to support different bayesian belief structures
###NOTE: what is x_sigma??? Need to understand theory and test everything!
class VAECriterion(nn.Module):
    """
    Here we calculate the VAE loss
    VAE loss's math formulation is :
    E_{z~Q}[log(P(X|z))]-D[Q(z|X)||P(z)]
    which can be transformed into:
    ||X-X_{reconstructed}||^2/(\sigma)^2 - [<L2norm(u)>^2+<L2norm(diag(\Sigma))>^2
    -<L2norm(diag(ln(\Sigma)))>^2-1]
    Our input is :
    x_sigma,x_reconstructed,x,z_mean,z_Sigma
    """
    def __init__(self, discrete_dim=None, x_sigma=1, bce_reconstruction=False):
        """
        :param discrete_dim: discrete latent variable or labels)
        :param x_sigma: **needs updating** used in reconstruction loss
        :param bce_reconstruction: used in reconstruction loss
        """
        super().__init__()
        self.x_sigma = x_sigma
        self.bce_reconstruction = bce_reconstruction
        if discrete_dim is not None or 0: # maybe this is where we inject prior bias? since this is trying to make uniform label dist...?
            self.disc_log_prior_param = torch.log(torch.tensor([1/discrete_dim for i in range(discrete_dim)]).view(1, -1).float().cuda())
        else:
            self.disc_log_prior_param = None

    def forward(self, x, x_reconstructed, z_mean, z_log_sigma, disc_log_alpha=None):
        """
        :param x: input and ground truth image
        :param x_reconstructed: reconstructed output from VAE, p(x|z) or p(x|z,y)
        :param z_mean: mean of continuous latent variable z_mu
        :param z_log_sigma: log stdev of cont latent variable z_s
        :param disc_log_alpha: param list for the discrete parameter (labels); log(a1),...,log(an)
        :return: reconstruction_loss, continuous_kl_loss, discrete_kl_loss
        """

        batch_size = x.size(0)
        # calculate reconstruction loss, sum in instance, mean in batch
        # use binary CE loss for reconstruction loss
        if self.bce_reconstruction:
            reconstruction_loss = F.binary_cross_entropy_with_logits(x_reconstructed, x, reduction="sum")/(batch_size * (self.x_sigma**2))
        else:
            reconstruction_loss = F.mse_loss(torch.sigmoid(x_reconstructed), x, reduction="sum") / (2*batch_size * (self.x_sigma**2))
        
        # calculate kl divergence of latent space
        z_mean_sq = z_mean*z_mean
        z_log_sigma_sq = 2*z_log_sigma
        z_sigma_sq = torch.exp(z_log_sigma_sq)
        continuous_kl_loss = 0.5*torch.sum(z_mean_sq + z_sigma_sq - z_log_sigma_sq - 1) / batch_size
        #NOTE: duplicate the 0.5 from M1 criterion to M2... see SHOT-VAE for source
        discrete_kl_loss = None
        if disc_log_alpha is not None:
            discrete_kl_loss = torch.sum(torch.exp(disc_log_alpha)*(disc_log_alpha - self.disc_log_prior_param)) / batch_size

        return reconstruction_loss, continuous_kl_loss, discrete_kl_loss
    
class M1Criterion(nn.Module):
    def __init__(self, x_sigma=1, bce_reconstruction=True):
        super().__init__()
        self.x_sigma = x_sigma
        self.bce_reconstruction = bce_reconstruction
    
    def forward(self, x, x_reconstructed, M1_mean, M1_log_sigma):
        batch_size = x.size(0)
        if self.bce_reconstruction:
            reconstruction_loss = F.binary_cross_entropy_with_logits(x_reconstructed, x, reduction="sum") / (batch_size)
        else:
            #print(torch.sigmoid(x_reconstructed))
            reconstruction_loss = F.mse_loss(torch.sigmoid(x_reconstructed), x, reduction="sum") / (2*batch_size*(self.x_sigma))

        # calculate kldiv of latent space
        M1_mean_sq = M1_mean*M1_mean
        M1_log_sigma_sq = 2*M1_log_sigma
        M1_sigma_sq = torch.exp(M1_log_sigma_sq)
        M1_continuous_kl_loss = 0.5*torch.sum(M1_mean_sq + M1_sigma_sq - M1_log_sigma_sq - 1) / batch_size

        return reconstruction_loss, M1_continuous_kl_loss
    
class M2Criterion(nn.Module): #WARNING! Assumes uniform random prior
    def __init__(self, discrete_dim=10):
        super().__init__()
        self.disc_log_prior_param = torch.log(torch.tensor([1/discrete_dim for i in range(discrete_dim)]).view(1,-1).float().cuda())

    def forward(self, M2_mean, M2_log_sigma, disc_log_alpha):
        batch_size = M2_mean.size(0)
        # calculate kldiv of latent space
        M2_mean_sq = M2_mean*M2_mean
        M2_log_sigma_sq = 2*M2_log_sigma
        M2_sigma_sq = torch.exp(M2_log_sigma_sq)
        M2_continuous_kl_loss = 0.5*torch.sum(M2_mean_sq + M2_sigma_sq - M2_log_sigma_sq - 1) / batch_size
        disc_kl_loss = torch.sum(torch.exp(disc_log_alpha) * (disc_log_alpha - self.disc_log_prior_param)) / batch_size
        return M2_continuous_kl_loss, disc_kl_loss

class ClsCriterion(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, predict, label, reduction='mean'):
        """
        :param predict: B*C log_softmax class predictions
        :param label: B*C one-hot label
        :param batch_weight: B*1 0-1 weight for each item in a batch
        :return: CE loss
        """

        if reduction == 'sum':
            cls_loss = -1*torch.sum(torch.sum(predict*label, dim=1))
        elif reduction == 'mean':
            cls_loss = -1*torch.mean(torch.sum(predict*label, dim=1))
        return cls_loss

class ReconstructionCriterion(nn.Module):
    #Calculate the -logp(x|z) criterion, in MSE or BCE forms
    
    def __init__(self, x_sigma=1, bce_reconstruction=True):
        super().__init__()
        self.x_sigma = x_sigma
        self.bce_reconstruction = bce_reconstruction

    def forward(self, x, x_reconstructed):
        batch_size = x.size(0)
        # calculate reconstruction loss, sum in instance, mean in batch
        # we use Binary Cross Entropy loss
        if self.bce_reconstruction:
            reconstruction_loss = F.binary_cross_entropy_with_logits(x_reconstructed, x, reduction="sum") / batch_size
        else:
            reconstruction_loss = F.mse_loss(torch.sigmoid(x_reconstructed), x, reduction="sum") / (2*batch_size*(self.x_sigma ** 2))
        return reconstruction_loss
    
class KLNormCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z_mean_pre, z_log_sigma_pre, z_mean_gt=None, z_sigma_gt=None):
        batch_size = z_mean_pre.size(0)
        if z_mean_gt is None or z_sigma_gt is None:
            """
            KL[N(z_mean_pre,z_sigma_pre)||N(0,I)]
            """
            z_mean_sq = z_mean_pre*z_mean_pre
            z_log_sigma_sq = 2 * z_log_sigma_pre
            z_sigma_sq = torch.exp(z_log_sigma_sq)
            kl_loss = 0.5 * torch.sum(z_mean_sq + z_sigma_sq - z_log_sigma_sq - 1) / batch_size
        else:
            """
            KL[N(z_mean_pre,z_sigma_pre)||N(z_mean_gt,z_sigma_gt)]
            """
            z_log_sigma_sq_pre = 2 * z_log_sigma_pre
            z_sigma_sq_pre = torch.exp(z_log_sigma_sq_pre)
            z_log_sigma_sq_gt = 2 * torch.log(z_sigma_gt + 1e-4)
            z_sigma_sq_gt = z_sigma_gt ** 2
            kl_loss = 0.5 * torch.sum(z_log_sigma_sq_gt - z_log_sigma_sq_pre + z_sigma_sq_pre / z_sigma_sq_gt + (
                    z_mean_pre - z_mean_gt) ** 2 / z_sigma_sq_gt - 1) / batch_size
        return kl_loss

class KLDiscCriterion(nn.Module):
    """
    sum (j=1,...,K) D_KL[q(c_j|x)||p(c_j|x)]
    """
    def __init__(self):
        super(KLDiscCriterion, self).__init__()

    def forward(self, disc_log_pre, disc_gt, qp_order=True):
        batch_size = disc_log_pre.size(0)
        disc_log_gt = torch.log(disc_gt + 1e-4)
        if qp_order:
            loss = torch.sum(torch.exp(disc_log_pre) * (disc_log_pre - disc_log_gt)) / batch_size
        else:
            loss = torch.sum(disc_gt * (disc_log_gt - disc_log_pre)) / batch_size
        return loss



