import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):            # image size = (4, 50, 224, 224), label size = (50, 224, 224)
    k = np.random.randint(0, 4)
    axis = np.random.randint(0, 2)

    for i in range(label.shape[0]):
        label[i] = np.rot90(label[i], k)
        label[i] = np.flip(label[i], axis=axis).copy()

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i][j] = np.rot90(image[i][j], k)
            image[i][j] = np.flip(image[i][j], axis=axis).copy()

    return image, label


def random_rotate(image, label):            # image size = (4, 50, 224, 224), label size = (50, 224, 224)
    angle = np.random.randint(-15, 15)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i][j] = ndimage.rotate(image[i][j], angle, order=0, reshape=False).copy()
    for i in range(label.shape[0]):
        label[i] = ndimage.rotate(label[i], angle, order=0, reshape=False).copy()
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']    # image size = (4, 50, 224, 224), label size = (50, 224, 224)     
        
        if random.random() > 0.6:
            image, label = random_rotate(image, label)
            image, label = random_rot_flip(image, label)
            
        x, y = image.shape[2:]
            
        if x != self.output_size[0] or y != self.output_size[1]:
            # order 0 = Nearest neighbor upsampling, fill with neighbor's value
            # order 1 = Nearest bilinear upsampling, order 2 = cubic
            for i in range(image):
                image[i] = zoom(image[i], (self.output_size[0] / x, self.output_size[1] / y), order=3)  
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class BraTS_dataset3D(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):   # list_dir = 'lists/list_BraTS/'
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()  
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            case_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, case_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

