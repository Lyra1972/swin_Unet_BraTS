import os
import cv2
import glob
import random
import numpy as np
from shutil import copyfile, move
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import transforms
from datasets import custom as tr

def read_own_data(root_path, split = 'train'):
    # root_path = "./Split_BraTS/train"
    images = []
    masks = []
    if split == 'train':
        t1 = glob.glob(r'./Split_BraTS/Train/*/*t1.nii.gz')
        t2 = glob.glob(r'./Split_BraTS/Train/*/*t2.nii.gz')
        flair = glob.glob(r'./Split_BraTS/Train/*/*flair.nii.gz')
        t1ce = glob.glob(r'./Split_BraTS/Train/*/*t1ce.nii.gz')
        seg = glob.glob(r'./Split_BraTS/Train/*/*seg.nii.gz')
    else:
        t1 = glob.glob(r'./Split_BraTS/Valid/*/*t1.nii.gz')
        t2 = glob.glob(r'./Split_BraTS/Valid/*/*t2.nii.gz')
        flair = glob.glob(r'./Split_BraTS/Valid/*/*flair.nii.gz')
        t1ce = glob.glob(r'./Split_BraTS/Valid/*/*t1ce.nii.gz')
        seg = glob.glob(r'./Split_BraTS/Valid/*/*seg.nii.gz')

    # 删
    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name) './train/images/3233'
        label_path = os.path.join(gt_root, image_name)

        images.append(image_path) 
        masks.append(label_path)

    return images, masks

# 删
def own_data_loader(img_path, mask_path):
    img = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path)
    mask = np.array(mask)
    mask[mask>0] = 1     #这里我把255转到了1
    mask = Image.fromarray(np.uint8(mask))
    return img, mask

class ImageFolder(data.Dataset):

    def __init__(self, args, split='train'):
        self.args = args
        self.root = self.args.root_path
        self.split = split
        self.images, self.labels = read_own_data(self.root, self.split)
    
    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.img_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            tr.ToTensor()
            ])
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.img_size),
            tr.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            tr.ToTensor()
            ])
        return composed_transforms(sample)

    def __getitem__(self, index):
        img, mask = own_data_loader(self.images[index], self.labels[index])
        if self.split == "train":
            sample = {'image': img, 'label': mask}
            return self.transform_tr(sample)
        elif self.split == 'val':
            img_name = os.path.split(self.images[index])[1]
            sample = {'image': img, 'label': mask}
            sample_ = self.transform_val(sample)
            sample_['case_name'] = img_name[0:-4]
            return sample_
        # return sample

    def __len__(self):
        assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)
