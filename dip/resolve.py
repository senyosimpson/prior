
import os
import random
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torchvision import transforms
from dip.utils import imresize
from dip.models.unet import UNet
from dip.datasets.set5 import Set5

if __name__ == '__main__':
    logger = logging.getLogger('dip')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        help='path to dataset')
    parser.add_argument('--logdir',
                        type=str,
                        required=True,
                        help='path to directory to save hr image outputs')
    parser.add_argument('--n-steps',
                        type=int,
                        default=10000,
                        required=False,
                        help='number of steps to take in optimization process')
    args = parser.parse_args()
    ds_path = args.dataset
    logdir = args.logdir
    steps = args.n_steps

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(input_channels=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    mse = nn.MSELoss()

    tsfm = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = Set5(ds_path, transform=tsfm)

    idx = random.choice(np.arange(len(dataset)))
    idx = 2
    lres_img = dataset[idx]
    lres_img = torch.unsqueeze(lres_img, dim=0)
    lres_h, lres_w = list(lres_img.shape)[2:]
    hres_h = lres_h * 4; hres_w = lres_w * 4
    noise = torch.zeros(1, 32, hres_h, hres_w)
    noise = noise.uniform_() * 1./10
    input_noise = noise.detach().clone()
    additive_noise = noise.detach().clone()

    lres_img = lres_img.to(device)
    for step in range(steps):
        z = input_noise + (additive_noise.normal_() * 1./30)
        z = z.to(device)
        optimizer.zero_grad()
        hr = model(z)
        hr = torch.squeeze(hr)
        gen_lr = imresize(hr, scale=0.25, device=device)
        gen_lr = torch.unsqueeze(gen_lr, dim=0)
        loss = mse(gen_lr, lres_img)
        loss.backward()
        optimizer.step()
        logger.info('step: [%d/%d], loss: %.5f' % (step+1, steps, loss.item()))

        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                output = model(z)
                output = output.squeeze()
                output = TF.to_pil_image(output.cpu())
                save_path = 'iteration-%d.png' % step
                save_path = os.path.join(logdir, save_path)
                output.save(save_path, 'PNG')
            model.train()
