import os
from PIL import Image
from torch.utils.data import Dataset
from glob import glob


class Set14(Dataset):
    def __init__(self, root, fmat='png', transform=None):
        ext = '*.%s' % fmat
        self.dataset = self.build_dataset(os.path.join(root, ext))
        self.transform = transform

    def build_dataset(self, root):
        image_paths = glob(root)
        dataset = sorted([path for path in image_paths if 'LR' in path])
        return dataset

    def __getitem__(self, idx):
        lres_img_path, hres_img_path = self.dataset[idx]
        lres_img = Image.open(lres_img_path)
        hres_img = Image.open(hres_img_path)
        # third image is grayscale, convert to RGB
        if idx == 2:
            lres_img = lres_img.convert('RGB')
            hres_img = hres_img.convert('RGB')
        sample = (lres_img, hres_img)

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.dataset)
