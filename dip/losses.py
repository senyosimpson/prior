import torch
import torch.nn as nn
from torchvision.models import vgg19


class VGGLoss(nn.Module):
    def __init__(self, layer=34):
        super().__init__()
        use_cuda = not False and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.mse = nn.MSELoss()
        vgg = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features.children()))[:layer+1]
        self.feature_extractor.to(self.device)
        for k, v in self.feature_extractor.named_parameters():
            v.requires_grad = False
    
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
    
    def forward(self, gen_hr, real_hr, beta=0.006):
        gen_hr = (gen_hr - self.mean) / self.std
        gen_hr_feat_map = self.feature_extractor(gen_hr)
        real_hr = (real_hr - self.mean) / self.std
        real_hr_feat_map = self.feature_extractor(real_hr)

        loss = beta *  self.mse(gen_hr_feat_map, real_hr_feat_map)
        return loss