from __future__ import annotations

import os, platform
import numpy as np
import pandas as pd
import pydicom
import cv2
from multiprocessing import Pool
from pydicom.pixel_data_handlers.util import apply_voi_lut

# courtesy of https://github.com/dungnb1333/SIIM-COVID19-Detection/blob/main/src/prepare/dicom2image_siim.py

# really naive solution...
def get_label_name(label_array):
    if label_array == [1,0,0,0]:
        return "negative"
    elif label_array == [0,1,0,0]:
        return "typical"
    elif label_array == [0,0,1,0]:
        return "indeterminate"
    elif label_array == [0,0,0,1]:
        return "atypical"
    else:
        print("unknown or multi-label label!")
        exit(1)

class ImageInstance:
    def __init__(self, uid, fpath, label):
        self.uid = uid
        self.fpath = fpath
        self.label = label
    
def dicom_to_img(instance:ImageInstance):
    # watch for split path diffs between os's
    delimiter = "\\" if platform.system() == 'Windows' else '/'
    image_id = instance.fpath.split(delimiter)[-1].split('.')[0]
    dcm_file = pydicom.dcmread(instance.fpath)

    # some checks, I assume?
    #patient_id = dcm_file.PatientID
    assert image_id == dcm_file.SOPInstanceUID
    #series_id = dcm_file.SeriesInstanceUID
    assert instance.uid == dcm_file.StudyInstanceUID

    # make pngs from dicoms. if VOI LUT available, transform it to more "human-friendly" view
    if dcm_file.get("VOILUTSequence"):
        data = apply_voi_lut(dcm_file.pixel_array, dcm_file) 
    else: data = dcm_file.pixel_array

    if dcm_file.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    # organization is as follows: 
    '''
    SIIM_train_images: {
        negative: {
            uid (study id): {
                imageid: Image
            }
        },
        ...
    }
    '''
    os.makedirs(f'./datasets/SIIM_train_images/{instance.label}/{instance.uid}', exist_ok=True)
    image_path = f'./datasets/SIIM_train_images/{instance.label}/{instance.uid}/{image_id}.png'
    
    res = cv2.imwrite(image_path, data)
    print(res)
if __name__ == '__main__':
    # dicom image folder. Since we only have labels for train (test hidden for competition), no merging is done
    dicom_path = './datasets/SIIM'
    assert os.path.exists(dicom_path)
    # folder to store all images. We will organize them by label for an imagefolder dataloader
    image_path = './datasets/SIIM_train_images'
    os.makedirs(str(image_path), exist_ok=True)

    instances = []

    df = pd.read_csv(dicom_path+'/train_study_level.csv')

    for id in np.unique(df.id.values):
        if '_study' not in id:
            continue
        uid = id.replace('_study', '')
        row = df.loc[df['id'] == id].values[0][1:]
        label = get_label_name(list(row))
        for dicom_dir, _, files in os.walk(dicom_path+'/train/'+uid):
            for file in files:
                fpath = os.path.join(dicom_dir, file)
                fname, f_ext = os.path.splitext(fpath)
                if f_ext in ['.dcm', '.dicom']:
                    instances.append(ImageInstance(uid=uid, fpath=fpath, label=label))

    p = Pool(16)
    results = p.map(func=dicom_to_img, iterable=instances)
    p.close()