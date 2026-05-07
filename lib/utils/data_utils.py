import numpy as np
from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
import os

# need to manually crop a couple edge cases... (6 total)
def _crop_US_image(img): # finds left and right edge by mode
    img_h, img_w = img.shape
    p_found = False
    left=-1
    for w_i in range(img_w):
        vline_sum = sum(img[:img_h//2,w_i])# only look at first half to avoid bottom text
        if vline_sum == 0: 
            if not p_found:
                #list_l.append(img[:,w_i])
                continue
        else:
            #if p_found == False: print('found non-zero pixel at', w_i)
            p_found = True
            #list_p.append(img[:,w_i+27])
            left = w_i + 27
            #print('left found at', left)
            break
            # test: is there at least a single line of only black pixels between (P) and US image
    if left == -1:
        raise IndexError
    loff = 10
    # find top left
    top_left=-1
    for w_i in range(img_w-left-loff):
        w = w_i + left + loff
        vline = img[:,w]
        for i, v_pixel in enumerate(vline):
            if v_pixel > 0:
                top_left = i
                break
        if top_left != -1:
            break
    if top_left == -1:
        raise IndexError
    
    # find right edge
    # keep going right from top_left until pixel==0
    right=-1
    voff = 10
    hoff = 10
    hstrip = img[:top_left+voff,left+hoff:] # img[**top_left**:top_left+voff,left+hoff:]
    #print(hstrip.shape)
    #print(hstrip[:,448])
    for i in range(hstrip.shape[1]):
        if sum(hstrip[:,i]) == 0:
            right=left+hoff+i
            break
    if right == -1:
        raise IndexError
    
    #print(left, top_left, right)

    new_image = img[top_left:,left:right]
    return new_image

def create_cropped_images():
    new_dpath = Path.cwd() / 'datasets' / 'hip734_cropped'
    os.makedirs(new_dpath, exist_ok=True)
    os.makedirs(new_dpath/'dysplastic', exist_ok=True)
    os.makedirs(new_dpath/'normal', exist_ok=True)
    imageFolder = ImageFolder(root=Path.cwd() / 'datasets' / 'hip734', transform=transforms.Grayscale())
    for image_i in range(len(imageFolder)):
        new_image = _crop_US_image(np.array(imageFolder[image_i][0]))
        label = imageFolder[image_i][1]
        im = Image.fromarray(new_image)
        imname = imageFolder.imgs[image_i][0].split(sep='\\')[-1]
        if label == 0: #dysplastic
            im.save(fp=new_dpath/'dysplastic'/imname)
        else: #normal
            im.save(fp=new_dpath/'normal'/imname)
    print("new dataset of cropped images created!")

if __name__ == '__main__':
    create_cropped_images()
    print("complete")