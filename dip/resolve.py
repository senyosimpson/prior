# Generate random noize z
# Pass through model that upsamples and super resolves image
# Downsample output
# Compare against LR image
# Update weights
# Run till convergence

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
from dip.config import Config
from dip.constants import *
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
    parser.add_argument('--config',
                        type=str,
                        required=True,
                        help='path to config file')
    opts = parser.parse_args()
    config = Config(opts.config)
    steps = config['steps']
    model_name = config['model']
    ds_path = config['dataset_path']
    logdir = config['logdir']

    use_cuda = not False and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = UNet().to(device)
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
    z = torch.zeros(1, 3, hres_h, hres_w)
    z = z.normal_() * 1./10
    #noise = z.clone().detach()

    lres_img.to(device)
    z.to(device)
    for step in range(steps):
        logger.info('Step [%d]/[%d]' % (step+1, steps))
        optimizer.zero_grad()
        hr = model(z)
        hr = torch.squeeze(hr)
        gen_lr = imresize(hr, scale=0.25)
        gen_lr = torch.unsqueeze(gen_lr, dim=0)
        loss = mse(gen_lr, lres_img)
        loss.backward()
        optimizer.step()
        logger.info('step: %d, loss: %.5f' % (step+1, loss.item()))
        #z = z + noise * 0.03

        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                output = model(z)
                output = output.squeeze()
                output = TF.to_pil_image(output)
                save_path = 'iteration-%d.png' % step
                save_path = os.path.join(logdir, save_path)
                output.save(save_path, 'PNG')
            model.train()
