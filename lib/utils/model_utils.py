import math
import torch
import os
from os import path

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def modify_lr_rate(opt, lr):
    for param_group in opt.param_groups:
        param_group['lr'] = lr

#NOTE: want something like this for kl-div beta, in which we want to relax latent space constraint until later on in training (ex: exponential growth of beta 1->5)
#NOTE: ref: https://github.com/edgarschnfld/CADA-VAE-PyTorch/issues/25
def alpha_schedule(offset, epoch, max_epoch, alpha_max): #NOTE: don't know what this is or why we do it...
    if epoch >= offset:
        alpha = alpha_max * math.exp(-5 * (1 - min(1, (epoch-offset) / (max_epoch))) ** 2)
    else:
        alpha = 0
    return alpha

def step_schedule(epoch, epoch_start, epoch_end, alpha_max):
    if epoch < epoch_start or epoch >= epoch_end:
        alpha = 0
    else:
        alpha = alpha_max
    return alpha

def pulse_schedule(offset, epoch, max_epoch, alpha_max, epoch_plateau):
    if epoch >= offset:
        if epoch < offset+max_epoch+epoch_plateau: alpha = alpha_max * math.exp(-5 * (1 - min(1, (epoch-offset) / max_epoch)) ** 2)
        else: alpha = alpha_max + (-1)*alpha_max * math.exp(-5 * (1 - min(1, (epoch - offset - max_epoch - epoch_plateau)/ max_epoch)) ** 2)
    else:
        alpha = 0
    return alpha

def beta_schedule(epoch, R, beta_min=1, beta_max=5):
    return min(beta_min*((1+R)**epoch), beta_max)

def save_checkpoint(output_dir, state, filename='checkpoint'):
    """
    :param state: a dict including:{
                'epoch': epoch + 1,
                'args': args,
                "state_dict": shot_vae_model.state_dict(),
                'optimizer': optimizer.state_dict(),
        }
    :param filename: the filename for store
    :param best_predict: the best predict flag
    :return:
    """
    filefolder = output_dir+'/checkpoints'
    if not path.exists(filefolder):
        os.makedirs(filefolder)
    torch.save(state, path.join(filefolder, filename+'.pth.tar'))

if __name__ == '__main__':
    filefolder = './output/SIIM_train_images/CNN_only/checkpoints/tr0.8_ar0.1_s0_k0_CNN_s0'
    fname = 'latest.pth.tar'
    print(path.join(filefolder, fname))