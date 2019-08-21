# Generate random noize z
# Pass through model that upsamples and super resolves image
# Downsample output
# Compare against LR image
# Update weights
# Run till convergence

import random
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from dip.config import Config
from dip.models.unet import UNet
from dip.datasets.set5 import Set5
from dip.utils import imresize

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
    ds_path = config['dataset']

    use_cuda = not False and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = UNet()
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

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
        logging.info('Step [%d]/[%d]' % (step+1, steps))
        optimizer.zero_grad()
        hr = model(z)
        hr = torch.squeeze(hr)
        gen_lr = imresize(hr, scale=0.25)
        gen_lr = torch.unsqueeze(gen_lr, dim=0)
        print(gen_lr.shape)
        print(lres_img.shape)
        loss = mse(gen_lr, lres_img)
        loss.backward()
        optimizer.step()
        logger.info('step: %d, loss: %.5f' % (step, loss.item()))
        #z = z + noise * 0.03
