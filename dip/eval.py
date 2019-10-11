import logging
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dip.datasets.evaluation import Set5
from dip.datasets.evaluation import Set14
from dip.metrics import PSNR
from dip.metrics import SSIM

if __name__ == '__main__':
    logger = logging.getLogger('dir')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        help='the dataset to evaluate model on')
    parser.add_argument('--root-dir',
                        type=str,
                        required=True,
                        help='the path to the root directory of the dataset')
    parser.add_argument('--batch-size',
                        type=int,
                        default=1,
                        required=False,
                        help='the batch size')
    parser.add_argument('--num-workers',
                        type=int,
                        default=4,
                        required=False,
                        help='number of workers used in loading data')
    args = parser.parse_args()
    root_dir = args.root_dir
    dataset = args.dataset
    batch_size = args.batch_size
    num_workers = args.num_workers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tsfm = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = Set14(root_dir, transform=tsfm)
    dataloader = DataLoader(dataset,
                            shuffle=False,
                            batch_size=batch_size,
                            num_workers=num_workers)

    psnr = PSNR()
    ssim = SSIM()

    logger.info('============== Starting Evaluation ==============')
    with torch.no_grad():
        avg_psnr = 0
        avg_ssim = 0
        for idx, image_pair in enumerate(dataloader):
            logger.info('Batch %d/%d' % (idx + 1, len(dataloader)))
            generated_image, hr_image = image_pair
            generated_image = generated_image.to(device)
            hr_image = hr_image.to(device)
            # crop border pixels
            hr_image = hr_image[:, :, 4:-4, 4:-4]
            generated_image = generated_image[:, :, 4:-4, 4:-4]
            # evaluate metrics
            avg_psnr += psnr(generated_image, hr_image)
            avg_ssim += ssim(generated_image, hr_image)

    print("Avg. PSNR: {:.4f} dB".format(avg_psnr / len(dataloader)))
    print("Avg. SSIM: {:.4f} dB".format(avg_ssim / len(dataloader)))
