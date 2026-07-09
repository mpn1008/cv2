import torch
import torch.nn as nn

from .decoder import SaliencyDecoder
from .encoder import ResNetEncoder


class SaliencyNet(nn.Module):
    """
    End-to-end fixation prediction network.

    Input : scene image  (B, 3, H, W)  – ImageNet-normalised
    Output: saliency map (B, 1, H, W)  – values in [0, 1]
    """

    def __init__(self, pretrained: bool = True, decoder_channels: list[int] = None):
        super().__init__()
        self.encoder = ResNetEncoder(pretrained=pretrained)
        self.decoder = SaliencyDecoder(decoder_channels=decoder_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        features = self.encoder(x)
        return self.decoder(features, target_size=(h, w))
