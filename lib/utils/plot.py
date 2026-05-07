from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

def plot_images(images, labels, normalize = True, fname='Images'):

    n_images = len(images)
    #print(n_images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))
    #print(rows, cols)

    fig = plt.figure(figsize = (15, 15))

    for i in range(rows*cols):

        ax = fig.add_subplot(rows, cols, i+1)
        
        image = images[i]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image.permute(1, 2, 0).cpu().numpy()) # C x H x W --> H x W x C for matplot
        if labels is not None:
            label = labels[i]
            ax.set_title(label)
        ax.axis('off')
    fig.savefig(f"{fname}.png")
    plt.close(fig)

def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min=image_min, max=image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5) #[0,1] range
    return image

# methods below are for visualizing transformer outputs

def show_image(image, mean=0.1642, std=0.1484, title=""):
    # image is [H, W, 3]?
    assert image.shape[2] == 3
    #image = normalize_image(image)
    plt.imshow(torch.clip((image * std + mean) * 255, 0, 255).int())
    plt.imshow(image)
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def visualize(image, outputs, model, batched=True, show_images=False):
    # forward pass
    logits = outputs.logits[0].unsqueeze(0) if batched else outputs.logits
    y = model.unpatchify(logits)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize mask
    mask = outputs.mask[0].unsqueeze(0).detach() if batched else outputs.mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size**2 *3) # (N, H*W, p*p*3) # for single batch, mask size = [1, 196, 768]
    mask = model.unpatchify(mask) # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', image)

    # masked image
    im_masked = x * (1-mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1-mask) + y*mask
    
    if not show_images:
        return im_masked[0], y[0], im_paste[0]
    else:
        # make figure larger
        plt.rcParams['figure.figsize'] = [20,20]

        plt.subplot(1,4,1)
        show_image(x[0], title="original")

        plt.subplot(1,4,2)
        show_image(im_masked[0], title="masked")

        plt.subplot(1,4,3)
        show_image(y[0], title="reconstruction")

        plt.subplot(1,4,4)
        show_image(im_paste[0], title="reconstruction + visible")

        plt.show()