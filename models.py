import torch
import torch.nn as nn
from i3d import InceptionI3d

I3D_PRETRAINED_PATH = "/workspace/MILForVideos/extractors/i3d/rgb_i3d_pretrained.pt"
class WrapperI3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.i3d = InceptionI3d()
        self.i3d.load_state_dict(torch.load(I3D_PRETRAINED_PATH))

    def forward(self, batches: torch.tensor):
        # B x Cls x F x C x H x W
        B = batches.size(0)
        Cls = batches.size(1)
        batches = batches.reshape(-1, *batches.size()[2:])
        ret = self._forward(batches)
        ret = ret.reshape(B, Cls, *ret.size()[1:])
        return ret
    
    def _forward(self, videos):
        videos = videos.transpose_(1, 2)
        return self.i3d(videos).squeeze(2)

class WrapperResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)

    def forward(self, batches: torch.tensor):
        # B x Cls x 1 x C x H x W
        B = batches.size(0)
        Cls = batches.size(1)
        batches = batches.reshape(-1, *batches.size()[2:])
        ret = self._forward(batches)
        ret = ret.reshape(B, Cls, *ret.size()[1:])
        return ret
        
    def _forward(self, images):
        images = images.squeeze(1)
        return self.resnet(images)