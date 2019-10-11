import os
from PIL import Image
from torch.utils.data import Dataset
from glob import glob


class Set5(Dataset):
    def __init__(self, root, fmat='png', transform=None):
        ext = '*.%s' % fmat
        self.dataset = self.build_dataset(os.path.join(root, ext))
        self.transform = transform

    def build_dataset(self, root):
        image_paths = glob(root)
        generated_img = sorted([path for path in image_paths if 'GEN' in path])
        hr_img = sorted([path for path in image_paths if 'HR' in path])
        dataset = list(zip(generated_img, hr_img))
        return dataset

    def __getitem__(self, idx):
        gen_img, hr_img = self.dataset[idx]
        gen_img = Image.open(gen_img)
        hr_img = Image.open(hr_img)
        if self.transform:
            gen_img = self.transform(gen_img)
            hr_img = self.transform(hr_img)
        sample = (gen_img, hr_img)
        return sample

    def __len__(self):
        return len(self.dataset)


class Set14(Dataset):
    def __init__(self, root, fmat='png', transform=None):
        ext = '*.%s' % fmat
        self.dataset = self.build_dataset(os.path.join(root, ext))
        self.transform = transform

    def build_dataset(self, root):
        image_paths = glob(root)
        generated_img = sorted([path for path in image_paths if 'GEN' in path])
        hr_img = sorted([path for path in image_paths if 'HR' in path])
        dataset = list(zip(generated_img, hr_img))
        return dataset

    def __getitem__(self, idx):
        gen_img_path, hr_img_path = self.dataset[idx]
        gen_img = Image.open(gen_img_path)
        hr_img = Image.open(hr_img_path)
        if '003' in gen_img_path:
            gen_img = gen_img.convert('RGB')
            hr_img = hr_img.convert('RGB')
        if self.transform:
            gen_img = self.transform(gen_img)
            hr_img = self.transform(hr_img)
        sample = (gen_img, hr_img)
        return sample

    def __len__(self):
        return len(self.dataset)
