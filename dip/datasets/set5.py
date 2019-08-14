import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from glob import glob


class Set5(Dataset):
    def __init__(self, root, fmat='png', colour_space='rgb', transform=None):
        ext = '*.%s' % fmat
        self.dataset = self.build_dataset(os.path.join(root, ext))
        self.transform = transform
        self.cspace = colour_space

    def build_dataset(self, root):
        image_paths = glob(root)
        lr_images = sorted([path for path in image_paths if 'LR' in path])
        hr_images = sorted([path for path in image_paths if path not in lr_images])
        dataset = list(zip(lr_images, hr_images))
        return dataset

    def __getitem__(self, idx):
        lres_img_path, hres_img_path = self.dataset[idx]
        lres_img = Image.open(lres_img_path)
        hres_img = Image.open(hres_img_path)
        if self.cspace == 'ycbcr':
            lres_img = lres_img.convert('YCbCr')
            hres_img = hres_img.convert('YCbCr')
        sample = (lres_img, hres_img)

        if self.transform:
            sample = self.transform(sample)
        return sample
        
    def __len__(self):
        return len(self.dataset)