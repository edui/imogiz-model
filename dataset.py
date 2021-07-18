import numpy as np
import cv2
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
#from keras.utils import to_categorical
from torch.autograd import Variable

"""
Implementation of Dataset Class for Height Estimation TRAIN images

    Input parameters:
        ds_dir: csv file which includes dataset samples.
        ds_name: Dataset name (csv file name).
        classify: Redundant, keep it False.
        
    Output:
        - Dictionary includes: input image, input mask, input joint locations
        and weight input.
"""

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3


class Images(Dataset):

    def __init__(self, ds_dir, ds_name, classify):
        self.df = pd.read_csv(ds_dir + ds_name, header=None)
        self.to_tensor = transforms.ToTensor()
        self.classify = classify
        self.ds_dir = ds_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        #dir_name = self.df.iloc[idx, 0]
        #image_name = dir_name+"/img.png"
        #mask_name = dir_name+"/label.png"

        image_name = self.df.iloc[idx, 0]
        mask_name = self.df.iloc[idx, 1]
        #joint_name = self.df.iloc[idx, 2]

        if self.classify:

            height = int(self.df.iloc[idx, 3])

            if height < 140:
                height = 0
            elif height >= 200:
                height = 13
            else:
                height = int((height-140)/5) + 1

        else:
            height = torch.from_numpy(
                np.array([self.df.iloc[idx, 3]/100])).type(torch.FloatTensor)

        # Reading Image
        img = cv2.imread(self.ds_dir + image_name)
        resized_img = cv2.resize(
            img, dsize=(IMG_HEIGHT, IMG_WIDTH))
            #, mode='constant', preserve_range=True)
        X = cv2.cvtColor(resized_img,
                         cv2.COLOR_BGR2RGB).astype('float32')
        X /= 255
        X = self.to_tensor(X)

        # Reading Mask
        mask = cv2.imread(self.ds_dir + mask_name)
        resized_mask = cv2.resize(
            mask, dsize=(IMG_HEIGHT, IMG_WIDTH))
            #, mode='constant', preserve_range=True)
        #y_mask = (cv2.imread(mask_name, 0) > 200).astype('float32')
        #y_mask = (resized_mask > 200).astype('float32')
        y_mask = cv2.cvtColor(resized_mask,
                              cv2.COLOR_BGR2RGB).astype('float32')
        #y_mask = to_categorical(y_mask, 2)
        y_mask = self.to_tensor(y_mask)

        # Reading Joint
        # y_heatmap = np.load(joint_name).astype('int64')  # For Heatmaps
        #y_heatmap = torch.from_numpy(y_heatmap)

        # Reading Height
        y_height = height

#        return {'img': X, 'mask': y_mask, 'joint': y_heatmap, 'height': y_height}
        return {'img': X, 'mask': y_mask,  'height': y_height}
