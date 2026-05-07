from __future__ import annotations

from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, datasets
import torch.utils.data as data
import torchvision.transforms.functional as F
import random
import numpy as np
import os, platform
import json
import math
#import matplotlib.pyplot as plt

# path by os
delimiter = '\\' if platform.system() == 'Windows' else '/'

class SquarePad():
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')

class MinMaxNorm():
    def __init__(self, min, max):
        self.min = min
        self.max = max
    def __call__(self, image):
        min_v = image.min()
        max_v = image.max()
        # Constant (e.g. blank padded) patches: max == min would divide by zero → NaNs
        # in the batch and break train/val loss. Epsilon yields a flat mid-range tensor.
        denom = (max_v - min_v).clamp(min=1e-8)
        new_image = (image - min_v) / denom * (self.max - self.min) + self.min
        return new_image

# crops aligned at top middle pixel to specified size, then resizes to desired output size
class ResizedCrop():
    def __init__(self, output_size):
        self.output_size = list(output_size)
    def __call__(self, image):
        print(image.size())
        h, w = image.size()
        left = (w//2)-(h//2) if w > h else 0
        crop_size = min(w,h)
        F.resized_crop(img=image, top=0, left=left, height=crop_size, width=crop_size, size=self.output_size, antialias=True)

# current iteration: very basic, only couple images are longer than wide and applies random zoom
class ResizedRandomCrop():
    # zoom is how many pixels in each side are excluded (akin to zooming in by n pixels)
    def __init__(self, output_size, max_zoom_percent=10, static=False):
        self.output_size = list(output_size)
        self.max_zoom_percent = max_zoom_percent
        self.static = static
    def __call__(self, image):
        #print(image.shape[-2:])
        h, w = image.shape[-2:]
        margin = max(w,h) - min(w,h) # top = 0 for w>h, left=0 for h>w
        #print(margin)
        if margin != 0:
            if w > h:
                top = 0
                left = np.random.randint(margin) if not self.static else 0
            else:
                left = 0
                top = np.random.randint(margin) if not self.static else 0
        else:
            top, left = (0,0)
        #print('left:', left, 'top:', top)
        zoom = np.random.randint(int(min(w,h)*((self.max_zoom_percent)/100))+1)
        #print('zoom:', str(zoom))
        crop_size = min(w,h)-zoom
        return F.resized_crop(img=image, top=top, left=left, height=crop_size, width=crop_size, size=self.output_size, antialias=True)

class HipDataset(Dataset):
    def __init__(self, subset, transform=None, pslabels=None):
        super().__init__()
        self.subset = subset
        self.pslabels = pslabels
        self.transform = transform

    def __getitem__(self, index):
        img, lbl = self.subset[index]
        #plt.imshow(img)
        #plt.show()
        if self.transform:
            img = self.transform(img)
        if self.pslabels:
            pslbl = self.pslabels[str(index)] 
            return img, lbl, pslbl
        else:
            return img, lbl, index #NOTE: return index, please refactor dataloader gets to work with this...

    def __len__(self):
        return len(self.subset)

# switching to generalized dataset class, might retire pseudolabel class-level cache...
class PatientDataset(Dataset): # DECISION TO TURN THIS TO BINARY: negative = 0, everything else = 1 as indeterminate is presence but not conclusive covid
    def __init__(self, subset, transform=None, dataset_type='Covid'):
        super().__init__()
        self.subset = subset
        self.transform = transform
        self.dataset_type = dataset_type

    def __getitem__(self, index):
        img, lbl = self.subset[index]
        #plt.imshow(img) # why won't this and prints work here? :(
        #plt.show()
        if self.dataset_type == 'SIIM_train_images': # atypical: 0, indet: 1, neg: 2, typical: 3 -> neg: 0, everything_else: 1
            lbl = 1 if lbl in [0, 1, 3] else 0
        if self.transform:
            img = self.transform(img)
        return img, lbl, index #NOTE: return index, please refactor dataloader gets to work with this...

    def __len__(self):
        return len(self.subset)

    def __call__(self, index):
        return self.subset[index]

# this needs to be per-split basis. grayscale transform is unnecessary here...
def normalize_dataset(dataset, hyperparams, grayscale=True, transform=None, write_config=True): 
    # not working with rgb images for now. NOTE WARNING! torch.flatten might need start-dim alignment for rgb imgs...
    means = torch.zeros(1) if grayscale else torch.zeros(3)
    stds = torch.zeros(1) if grayscale else torch.zeros(3)
    print(f"normalizing dataset of len {len(dataset)}")
    for i, (img, _) in enumerate(dataset): # i assume _ is num_channels?
        # debug line, takes so long
        if i % 100 == 0: print(f'normalizing image {i}...')
        # apply grayscale and to_tensor transforms
        if transform:
            img = transform(img)
        img = torch.flatten(img)
        means += torch.mean(img)
        stds += torch.std(img)

    means /= len(dataset)
    stds /= len(dataset)
    print(f'Calculated means: {means}')
    print(f'Calculated stds: {stds}')

    if write_config:
        dataset_name = hyperparams['dataset_name']
        vr = hyperparams['vr']
        s = hyperparams['s']
        tr = hyperparams['tr']
        k = hyperparams['k']
        config_path = f'./PT_configs/{dataset_name}'
        fpath = f'{config_path}/tr{tr}_vr{vr}_s{s}.json'
        if not os.path.exists(config_path):
            os.makedirs(config_path)
        if os.path.exists(fpath):
            with open(file=fpath, mode="r") as f:
                PT_config = json.load(f)
            PT_config[str(k)] = [str(means.item()),str(stds.item())]
        else: PT_config = {str(k):[str(means.item()),str(stds.item())]}
        with open(file=fpath, mode='w+') as f:
            json.dump(PT_config, f)

    return means, stds

def get_transform(dataset, dataset_type, hyperparams, image_size, PT_config, num_output_channels, normalize=True):
    # gets the respective transform for either train or valid/test 
    # base transform, grayscale and make tensor
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels),
        transforms.ToTensor(), # 0-255 -> 0-1
    ])

    # need to do normalization on per-split basis... thus will need to capture seed, [mean, std], and k to replicate norm params
    if normalize: #
        if PT_config is not None: mean, std = PT_config
        else: 
            # dataset is scoped by train/test split, but NOTE: we only normalize training data
            mean, std = normalize_dataset(dataset, hyperparams, grayscale=True, transform=transform)
            print("computed mean is", str(mean), 'and std is', str(std))
            return [float(mean), float(std)]
        transform = transforms.Compose([
            transform, 
            transforms.Normalize(mean=mean, std=std), # can have negatives
            MinMaxNorm(min=0, max=1), # set it to correct pixel value range
        ])

    # after universal transforms, now comes train/test-specific transforms
    if dataset_type == 'train':
        train_transform = transforms.Compose([
                transform,
                #transforms.RandomHorizontalFlip(), #NOTE: Does this make diagnostic sense? Would we get a mirrored image of the hip (ie: left vs right?)
                ResizedRandomCrop(image_size, max_zoom_percent=10, static=False), # zoom, will have guaranteed black side edges if 0
                #transforms.GaussianBlur(kernel_size=(5,5), sigma=(0.1,1.5)), # NOTE: add rotations by +-5 degrees (will also blur due to interpolation) (experiments todo: w/o blur, and w/ rotations)
                #transforms.RandomRotation(5)
        ])
        return train_transform
    
    elif dataset_type == 'test' or 'valid':
        test_transform = transforms.Compose([
                transform,
                ResizedRandomCrop(image_size, max_zoom_percent=0, static=True) # zoom, will have guaranteed black side edges if 0
        ])
        return test_transform

def get_dataset(dataset_path, class_sizes=None, train_ratio=0.8, valid_ratio=0.2, keep_full=False, k_folds=0, k_fold_index=None, keep_remainder=False):
    # image_size: (660, 925) (H x W)
    """ 
    :return: train_valid, test datasets, or full_dataset, train_transfroms, test_transforms for full dataset
    """
    full_dataset = datasets.ImageFolder(root=dataset_path)
    if keep_full: return full_dataset
    else:
        if keep_remainder:
            train_dataset, valid_dataset, test_dataset, train_patients, train_plus_dataset = split_by_patient(dataset=full_dataset, class_sizes=class_sizes, train_ratio=train_ratio, valid_ratio=valid_ratio, keep_train=True, k_folds=k_folds, k_fold_index=k_fold_index, keep_remainder=keep_remainder)
            return train_dataset, valid_dataset, test_dataset, train_patients, train_plus_dataset
        else:
            train_dataset, valid_dataset, test_dataset, train_patients = split_by_patient(dataset=full_dataset, class_sizes=class_sizes, train_ratio=train_ratio, valid_ratio=valid_ratio, keep_train=True, k_folds=k_folds, k_fold_index=k_fold_index, keep_remainder=keep_remainder)
            return train_dataset, valid_dataset, test_dataset, train_patients

def get_ssl_labels(labels, annot_ratio, num_classes, label_indices=None):
    """
    :param labels: torch.array(int tensor) ex: [0,1,0,0,1,0,...] (aligned to image indices)
    :param annotated_num_per_class: the number of annotation we use for each classes
    :param num_classes: total number of classes
    :return: labels_train_u, labels_train_l, labels_valid
    """

    num_annotated = int(len(label_indices)*annot_ratio)
    if label_indices is not None:
        #label_indices_valid = []
        label_indices_train_u = []
        label_indices_train_l = []
        # NOTE: removed per-class num_annotated, just hoping that randperm gives reasonable label dist
        #for i in range(num_classes):
        loc = torch.nonzero(labels == i).view(-1) #ex: i=0 will return location of all entries where labels=0
        #label_indices_valid.extend(label_indices[loc[:valid_num_per_class]].tolist()) # doesn't do anything, but keep for consistency for now
        label_indices_train_l.extend(label_indices[loc[:num_annotated]].tolist())
        #unlabeled set is now separate from labeled
        label_indices_train_u.extend(label_indices[loc[num_annotated:]].tolist())
    
    else:
        #label_indices_valid = []
        label_indices_train_u = []
        label_indices_train_l = []
        for i in range(num_classes):
            loc = torch.nonzero(labels == i).view(-1)
            #do random perm to make sure uniform sample
            loc = loc[torch.randperm(loc.size(0))] #AL: indices need to be selected for (AND TESTED!)
            #label_indices_valid.extend(loc[:valid_num_per_class].tolist())
            label_indices_train_l.extend(loc[:num_annotated].tolist())
            #unlabeled labels also includes the train_labeled part, ignored in train
            label_indices_train_u.extend(loc[num_annotated:].tolist())

    print(f"unlabeled: {len(label_indices_train_u)}, labeled: {len(label_indices_train_l)}" )
    return label_indices_train_u, label_indices_train_l

def create_dataloaders(dataset_name, dataset_path, class_sizes=None, seed=0, train_ratio=0.7, valid_ratio=0.1, annot_ratio=0.5, batch_size=1, num_workers=1, num_output_channels=1, image_size=(224,224), k_folds=0, k_fold_index=None, keep_remainder=False):
    # parent function used to create dataloaders for train and test loops
    # supply PT config via... configparser? gotta learn initially though
    # TODO!!! Need to clarify if norm is done correctly (by split) and also that test/valid should not receive norm (cause we don't know beforehand right?)
    ssl = False if annot_ratio == 1.0 or 0.0 else True
    # NOTE: NEED CLASS SIZES TO BE THE SAME, NOT TRACKED!!!
    #''' # currently only have support for grayscale. rgb does not work with normalization...
    hyperparams = {'dataset_name': dataset_name, 'tr':train_ratio, 'vr':valid_ratio, 's':seed, 'k':k_fold_index}
    # Seed before any split / PT_config learning so all entry points match (e.g. cold PT cache).
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    PT_config = get_PT_config(**hyperparams)
    if PT_config is None: # really bad reroute to prevent test dataset from being normalized
        if keep_remainder: 
            train_dataset, valid_dataset, test_dataset, train_patients, train_plus_dataset = get_dataset(dataset_path=dataset_path, class_sizes=class_sizes, train_ratio=train_ratio, valid_ratio=valid_ratio, keep_full=False, k_folds=k_folds, k_fold_index=k_fold_index, keep_remainder=keep_remainder)
        else: train_dataset, valid_dataset, test_dataset, train_patients = get_dataset(dataset_path=dataset_path, class_sizes=class_sizes, train_ratio=train_ratio, valid_ratio=valid_ratio, keep_full=False, k_folds=k_folds, k_fold_index=k_fold_index, keep_remainder=keep_remainder)
        PT_config = get_transform(train_dataset, 'train', hyperparams=hyperparams, image_size=image_size, PT_config=PT_config, num_output_channels=num_output_channels, normalize=True)

    if keep_remainder: 
        train_dataset, valid_dataset, test_dataset, train_patients, train_plus_dataset = get_dataset(dataset_path=dataset_path, class_sizes=class_sizes, train_ratio=train_ratio, valid_ratio=valid_ratio, keep_full=False, k_folds=k_folds, k_fold_index=k_fold_index, keep_remainder=keep_remainder)
    else: train_dataset, valid_dataset, test_dataset, train_patients = get_dataset(dataset_path=dataset_path, class_sizes=class_sizes, train_ratio=train_ratio, valid_ratio=valid_ratio, keep_full=False, k_folds=k_folds, k_fold_index=k_fold_index, keep_remainder=keep_remainder)
    train_transform = get_transform(train_dataset, 'train', hyperparams=hyperparams, image_size=image_size, PT_config=PT_config, num_output_channels=num_output_channels, normalize=True)
    test_transform = get_transform(test_dataset, 'test', hyperparams=hyperparams, image_size=image_size, PT_config=PT_config, num_output_channels=num_output_channels, normalize=True)  
    #valid_indices = valid_dataset.indices
    valid_custom_dataset = PatientDataset(valid_dataset, test_transform, dataset_type=dataset_name)
    valid_dloader = DataLoader(valid_custom_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    #test_indices = test_dataset.indices
    test_custom_dataset = PatientDataset(test_dataset, test_transform, dataset_type=dataset_name)
    test_dloader = DataLoader(test_custom_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)

    if not ssl:
        train_custom_dataset = PatientDataset(train_dataset, train_transform, dataset_type=dataset_name)
        train_dloader = DataLoader(train_custom_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)
        if keep_remainder:
            train_plus_custom_dataset = PatientDataset(train_plus_dataset, train_transform, dataset_type=dataset_name)
            train_plus_dloader = DataLoader(train_plus_custom_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)
            return train_dloader, valid_dloader, test_dloader, train_plus_dloader
        else: return train_dloader, valid_dloader, test_dloader
    else:
        train_u_dataset, train_l_dataset = split_by_patient(train_dataset.dataset, class_sizes=None, train_ratio=annot_ratio, valid_ratio=0.0, keep_train=False, all_patients=train_patients, k_folds=0, k_fold_index=None, ssl=ssl)
        print(f"ssl split: unlabeled = {len(train_u_dataset)}, labeled = {len(train_l_dataset)}")
        train_dataset_u = PatientDataset(train_u_dataset, train_transform, dataset_type=dataset_name)
        train_dloader_u = DataLoader(train_dataset_u, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)
        # labeled dataset remains the same
        train_dataset_l = PatientDataset(train_l_dataset, train_transform, dataset_type=dataset_name)
        train_dloader_l = DataLoader(train_dataset_l, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)
        return train_dloader_u, train_dloader_l, valid_dloader, test_dloader

def get_PT_config(dataset_name, tr, vr, s, k):
    try:
        with open(f'./PT_configs/{dataset_name}/tr{tr}_vr{vr}_s{s}.json', 'r') as f:
            PT_config_dict = json.load(f)
            f.close()
        if str(k) in PT_config_dict.keys(): 
            PT_config = [float(n) for n in PT_config_dict[str(k)]]
            print(f"using PT_config for s={s}, tr={tr}, vr={vr}]: {PT_config}")
            return PT_config
        else: raise KeyError
    except:
        print("Error: dataset config does not have pre-trained values! Caching now...")
        return None
    print(f"PT_config for k={k} is not found. Learning now...")
    return None

def add_to_patient_dict(dataset_name, img_id, pid, label, pat_dict, sizes, counts, unique_only=True):
    if dataset_name == 'SIIM_train_images': # atypical: 0, indet: 1, neg: 2, typical: 3 -> neg: 0, everything_else: 1
            label = 1 if label in [0, 1, 3] else 0
    if label == 0 and counts[label] < sizes[label]:
        if pid in pat_dict.keys():
            if not unique_only: 
                pat_dict[pid].append(img_id) 
                counts[label] += 1 
        else:
            pat_dict[pid] = [img_id]
            counts[label] += 1
        return True
    elif label == 1 and counts[label] < sizes[label]:
        if pid in pat_dict.keys():
            if not unique_only:
                pat_dict[pid].append(img_id)
                counts[label] += 1
        else:
            pat_dict[pid] = [img_id]
            counts[label] += 1
        return True
    else:
        return False

def split_by_patient(dataset, class_sizes=None, train_ratio=0.7, valid_ratio=0.1, keep_train=False, all_patients=None, k_folds=5, k_fold_index=None, ssl=False, keep_remainder=False):
    ### NOTE: MANUAL INDEX SELECTION OF PATIENT NAME IN PATH!! Simple, unique-only datasets will have -1, non-unique-only datasets will have -2 (assuming pat_id/img_id.png)
    # splits dataset by unique patient instead of by image (generic, subfolders for patients, some have more than one image)
    # NOTE: train_ratio gets used as annotation ratio in the ssl splits! thus, n_train for ssl is actually n_unlabeled (hence len(all_patients)*(1-train_ratio))
    dataset_name = dataset.root.split(delimiter)[-1]

    manual_flag_hip = False
    if dataset_name == 'Covid':
        unique_only = True
        pid_index = -1
    elif dataset_name == 'Cancer':
        unique_only = True
        pid_index = -2
    elif dataset_name == 'hip734_cropped':
        unique_only = False
        manual_flag_hip = True
    else:
        # Default: patient id = image filename; at most one image per PID (e.g. SIIM_train_images).
        unique_only = True
        pid_index = -1

    lbl_counts = [0]*len(class_sizes) if class_sizes is not None else None
    init_pass = True if all_patients is None else False
    if all_patients is None:
        all_patients = {}
        remainder = {}
        for i, (img_path, label) in enumerate(dataset.imgs): #imgs is a list of tuples... indices are ordered
            if not manual_flag_hip: PID = img_path.split(delimiter)[pid_index]
            else: 
                image_path = img_path.split(delimiter)[-1]
                PID = image_path.split(' ')[0][:4] # MoJa, for example, is a unique patient.
            if class_sizes is not None:
                if not add_to_patient_dict(dataset_name, img_id=i, pid=PID, label=label, pat_dict=all_patients, sizes=class_sizes, counts=lbl_counts, unique_only=unique_only): # will only go here after lbl_counts is maxed out. Adding test cases now
                    if keep_remainder:
                        add_to_patient_dict(dataset_name, img_id=i, pid=PID, label=label, pat_dict=remainder, sizes=[math.inf]*len(class_sizes), counts=[0]*len(class_sizes), unique_only=unique_only)
                    else:
                        continue
                else:
                    continue
            else:
                if PID in all_patients.keys():
                    all_patients[PID].append(i)
                else:
                    all_patients[PID] = [i]
    unique_patients = len(all_patients)
    #NOTE: NEED TO CHECK THIS LOGIC!!!
    if k_fold_index is None: # really only used to split labeled and unlabeled. 
        n_train = int(unique_patients*train_ratio) if init_pass else int(unique_patients*(1-train_ratio)) 
        perm_patients = np.random.permutation([patient for patient in all_patients])
        train_patients = {patient:all_patients[patient] for patient in perm_patients[:n_train]}
        train_patient_indices = [i for ii in train_patients.values() for i in ii]
        # calculate how to then split the final partition into validate and test
        valid_patients = {patient:all_patients[patient] for patient in perm_patients[n_train:]}
        valid_patient_indices = [i for ii in valid_patients.values() for i in ii]
        test_patient_indices = []
    else: 
        partition_size = int(unique_patients / k_folds)
        perm_patients = np.random.permutation([patient for patient in all_patients]) #added random permute, ensure this is seed-controlled
        train_patients = {}
        valid_patients = {}
        n_valid = int((valid_ratio/(train_ratio+valid_ratio))*partition_size)
        for k in range(k_folds):
            if k == k_fold_index: # test patients is the kth partition. 
                if k == k_folds - 1: 
                    test_patients = {patient:all_patients[patient] for patient in perm_patients[k*partition_size:]}
                else: 
                    test_patients = {patient:all_patients[patient] for patient in perm_patients[k*partition_size:(k+1)*partition_size]}
            else: 
                if k == k_folds - 1: 
                    train_valid_patients = perm_patients[k*partition_size:]
                    train_patients.update({patient:all_patients[patient] for patient in train_valid_patients[n_valid:]})
                    valid_patients.update({patient:all_patients[patient] for patient in train_valid_patients[:n_valid]})
                else: 
                    train_valid_patients = perm_patients[k*partition_size:(k+1)*partition_size]
                    train_patients.update({patient:all_patients[patient] for patient in train_valid_patients[n_valid:]})
                    valid_patients.update({patient:all_patients[patient] for patient in train_valid_patients[:n_valid]})
        train_patient_indices = [i for ii in train_patients.values() for i in ii]
        valid_patient_indices = [i for ii in valid_patients.values() for i in ii]
        test_patient_indices = [i for ii in test_patients.values() for i in ii]
        if keep_remainder:
            remainder_indices = [i for ii in remainder.values() for i in ii]
            train_plus_dataset = data.Subset(dataset, remainder_indices+train_patient_indices)
            

    assert len(set(train_patient_indices).intersection(set(test_patient_indices))) == 0
    assert len(set(train_patient_indices).intersection(set(valid_patient_indices))) == 0
    assert len(set(valid_patient_indices).intersection(set(test_patient_indices))) == 0

    train_dataset = data.Subset(dataset, train_patient_indices)
    valid_dataset = data.Subset(dataset, valid_patient_indices)
    if len(test_patient_indices) != 0:
        test_dataset = data.Subset(dataset, test_patient_indices)
    else: test_dataset = []
    print(f"Split by patient successful: n_train: {len(train_dataset)}, n_valid: {len(valid_dataset)}, n_test: {len(test_dataset)}")
    if keep_train:
        if keep_remainder: return train_dataset, valid_dataset, test_dataset, train_patients, train_plus_dataset
        else: return train_dataset, valid_dataset, test_dataset, train_patients  
    else:
        if not ssl:
            if keep_remainder: return train_dataset, valid_dataset, test_dataset, train_plus_dataset
            else: return train_dataset, valid_dataset, test_dataset
        else: 
            # valid is actually labeled dataset split
            return train_dataset, valid_dataset

def k_fold_CV_indices(dataset_size, k=5):
    """
    :param dataset_size: length of dataset
    :param k: number of partitions
    :return: two list of indices, each as k train/test indices in order
    """
    # find hyperparameters like prior weight using cross validation
    # NOTE: needs to be done at patient level, not by image! (split by patient manually here)
    ratio = 1/k
    partition_size = dataset_size * ratio
    train_indices = []
    test_indices = []
    for kth in range(k):
        train_ll = 0
        valid_l = int(kth*partition_size)
        train_lr = valid_l
        valid_r = int((kth+1)*partition_size)
        train_rl = valid_r
        train_rr = dataset_size
        valid_indexes = list(range(valid_l, valid_r))
        random.shuffle(valid_indexes) # in-place
        test_indices.append(valid_indexes)
        trainl_indices = list(range(train_ll, train_lr))
        random.shuffle(trainl_indices)
        trainr_indices = list(range(train_rl, train_rr))
        random.shuffle(trainr_indices)
        train_indices.append(trainl_indices + trainr_indices)
    return train_indices, test_indices

def seed_worker(worker_id): # from https://pytorch.org/docs/stable/notes/randomness.html 
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataset_labels(tr=0.6, vr=0.2, ar = 1.0, ssl=False, k_folds=5, k=0, seed=0, label_type='train', device='cuda'):
    ### TODO: NEEDS STRICT TESTING!!! MAKE SURE PSEUDOLABELS ARE CORRECTLY INDEXED 
    labels = {}

    # randomization control for dataset
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    if not ssl: train_dloader, valid_dloader, test_dloader = create_dataloaders(dataset_name='Covid', dataset_path='./datasets/Covid', train_ratio=tr, valid_ratio=vr, annot_ratio=ar, ssl=ssl, k_folds=k_folds, k_fold_index=k)
    else: train_dloader_u, train_dloader_l, valid_dloader, test_dloader = create_dataloaders(dataset_name='Covid', dataset_path='./datasets/Covid', train_ratio=tr, valid_ratio=vr, annot_ratio=ar, ssl=ssl, k_folds=k_folds, k_fold_index=k)
    
    if label_type == 'train':
        dloader = train_dloader
    elif label_type == 'test':
        dloader = test_dloader
    elif label_type == 'train_u':
        dloader = train_dloader_u
    elif label_type == 'train_l':
        dloader = train_dloader_l
    else:
        print(f'label type {label_type} is not currently supported]')

    print(f"saving labels by index")
    print(f"dataset size: {len(dloader)}")

    with torch.no_grad():
        for image, label, index in dloader:
            labels[str(index.item())]=label.item()

    labels_path = f'./datasets/hip734_cropped_labels/'
    fname = f'{label_type}_tr{tr}_vr{vr}_ar{ar}_k{k}_s{seed}'
    if not os.path.exists(labels_path):
        os.makedirs(labels_path)
    with open(labels_path+f'{fname}.json', 'w+') as f:
        json.dump(labels, f)
    print("pseudolabel set creation complete. Can be found at", labels_path)

    return labels

def split_by_patient_hip734(dataset, train_ratio=0.7, valid_ratio=0.1, unique_patients=6054, keep_train=False, all_patients=None, k_folds=0, k_fold_index=None, ssl=False):
    # splits dataset by unique patient instead of by image (only for hip734 dataset)
    # train ratio will generally be lower on average
    test_ratio = 1 - train_ratio - valid_ratio
    n_train = int(unique_patients*train_ratio) if all_patients is None else int(len(all_patients)*(1-train_ratio))
    if all_patients is None:
        all_patients = {}
        for img_i, img in enumerate(dataset.imgs): #imgs is a list of tuples... indices are ordered
            image_path = img[0]
            #image_label = img[1]
            PID = image_path.split(sep=' ')[0].split('\\')[-1]
            if len(PID) > 8: # on mac/linux, path needs pruning again
                PID = PID.split(sep='/')[-1]
            if PID in all_patients.keys():
                all_patients[PID].append(img_i)
            else:
                all_patients[PID] = [img_i]
    # NOTE NOTE NOTE NOTE technically in production you would need train_labels + validation_labels amount of labels to accurately do this
    # So if I wish to claim performance on par with only p% labels, p must be (n_train+n_valid)/len(dataset) 
    # NOTE: randomization done here. No need to repeat in ssl...
    ratio = valid_ratio / (valid_ratio+test_ratio)
    if k_fold_index is None: # really only used to split labeled and unlabeled. valid is empty
        perm_patients = np.random.permutation([patient for patient in all_patients])
        train_patients = {patient:all_patients[patient] for patient in perm_patients[:n_train]}
        train_patient_indices = [i for ii in train_patients.values() for i in ii]
        # calculate how to then split the final partition into validate and test
        test_patients = {patient:all_patients[patient] for patient in perm_patients[n_train:]}
        test_patient_indices = [i for ii in test_patients.values() for i in ii]
        valid_patient_indices = []
    else: 
        perm_patients = np.random.permutation([patient for patient in all_patients]) #added random permute, ensure this is seed-controlled
        partition_size = int(unique_patients / k_folds)
        valid_partition = int(partition_size*ratio)
        train_patients = {}
        for k in range(k_folds):
            if k == k_fold_index: 
                if k == k_folds - 1: 
                    valid_patients = {patient:all_patients[patient] for patient in perm_patients[k*partition_size:(k*partition_size)+(valid_partition)]}
                    test_patients = {patient:all_patients[patient] for patient in perm_patients[(k*partition_size)+(valid_partition):]}
                else: 
                    valid_patients = {patient:all_patients[patient] for patient in perm_patients[k*partition_size:(k*partition_size)+(valid_partition)]}
                    test_patients = {patient:all_patients[patient] for patient in perm_patients[(k*partition_size)+(valid_partition):(k+1)*partition_size]}
            else: 
                if k == k_folds - 1: 
                    train_patients.update({patient:all_patients[patient] for patient in perm_patients[k*partition_size:]})
                else: 
                    train_patients.update({patient:all_patients[patient] for patient in perm_patients[k*partition_size:(k+1)*partition_size]})
        train_patient_indices = [i for ii in train_patients.values() for i in ii]
        valid_patient_indices = [i for ii in valid_patients.values() for i in ii]
        test_patient_indices = [i for ii in test_patients.values() for i in ii]

    assert len(set(train_patient_indices).intersection(set(test_patient_indices))) == 0
    assert len(set(train_patient_indices).intersection(set(valid_patient_indices))) == 0
    assert len(set(valid_patient_indices).intersection(set(test_patient_indices))) == 0

    train_dataset = data.Subset(dataset, train_patient_indices)
    test_dataset = data.Subset(dataset, test_patient_indices)
    if len(valid_patient_indices) != 0:
        valid_dataset = data.Subset(dataset, valid_patient_indices)
    else: valid_dataset = []
    print(f"Split by patient successful: n_train: {len(train_dataset)}, n_valid: {len(valid_dataset)}, n_test: {len(test_dataset)}")
    if keep_train:
        return train_dataset, valid_dataset, test_dataset, train_patients
    else:
        if not ssl:
            return train_dataset, valid_dataset, test_dataset
        else: 
            # lazy combine at the end to save code
            test_valid_dataset = data.Subset(dataset, test_patient_indices+valid_patient_indices)
            return train_dataset, test_valid_dataset


    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    if train_flag:
        transform = transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transform
        ])
    if train_flag:
        dataset = datasets.SVHN(root=dataset_base_path, split='train', transform=transform, download=True)
    else:
        dataset = datasets.SVHN(root=dataset_base_path, split='test', transform=transform, download=True)
    return dataset


if __name__ == "__main__":
    tr = 0.7
    vr = 0.1
    ar = 1.0
    seed = 42
    ssl = False
    label_type = 'train'
    k_folds = 5

    for k in range(k_folds):
        get_dataset_labels(tr=tr, vr=vr, ar=ar, ssl=ssl, k_folds=k_folds, k=k, seed=seed, label_type=label_type)
    
