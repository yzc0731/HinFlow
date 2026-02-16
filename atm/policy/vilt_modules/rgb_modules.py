import torch
import torch.nn as nn

class PatchEncoder(nn.Module):
    """
    A patch encoder that does a linear projection of patches in a RGB image.
    """

    def __init__(
        self, input_shape, patch_size=[16, 16], embed_size=64, no_patch_embed_bias=False
    ):
        super().__init__()
        C, H, W = input_shape
        num_patches = (H // patch_size[0] // 2) * (W // patch_size[1] // 2)
        self.img_size = (H, W)
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.h, self.w = H // patch_size[0] // 2, W // patch_size[1] // 2

        self.conv = nn.Sequential(
            nn.Conv2d(
                C, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            ),
            nn.BatchNorm2d(
                64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(
            64,
            embed_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False if no_patch_embed_bias else True,
        )
        self.bn = nn.BatchNorm2d(embed_size)
        # register all BN layers for stage-wise control
        self.bn_layers = []
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.bn_layers.append(module)
        self.bn_layers = list(set(self.bn_layers))

    def set_bn_stage(self, online: bool = False):
        """
        set BN layers to different modes based on the training stage
        """
        for bn in self.bn_layers:
            if online:
                bn.eval()
            else:
                bn.train()

    def forward(self, x, online: bool = False):
        if self.training:
            self.set_bn_stage(online)

        B, C, H, W = x.shape
        x = self.conv(x)
        x = self.proj(x)
        x = self.bn(x)
        return x
