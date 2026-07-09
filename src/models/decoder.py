import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import EncoderFeatures


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    """
    Upsample × 2 → concatenate skip → two ConvBnRelu.

    in_ch  : channels coming from the deeper decoder stage
    skip_ch: channels of the skip connection from the encoder
    out_ch : output channels
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SaliencyDecoder(nn.Module):
    """
    Decodes EncoderFeatures back to the original spatial resolution.

    Architecture (for ResNet-50 input 224×224):
        s5  2048        → block4 →  256   (7  → 14)
        s4  1024  skip  → block3 →  128   (14 → 28)
        s3   512  skip  → block2 →   64   (28 → 56)
        s2   256  skip  → block1 →   32   (56 → 56, same resolution as s2)
        s1    64  skip  → block0 →   16   (56 → 112)
                         upsample × 2 → 224
                         head (1×1 conv + sigmoid)

    decoder_channels: list of 4 output channel widths for blocks 4..1
                      (block0 is fixed at half of the last value)
    """

    def __init__(self, decoder_channels: list[int] = None):
        super().__init__()
        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32]

        # encoder skip channels (ResNet-50)
        enc_chs = (64, 256, 512, 1024, 2048)  # s1..s5

        d = decoder_channels  # [256, 128, 64, 32]

        # deepest block: bottleneck → d[0]  (no skip from encoder at this stage)
        self.bottleneck = nn.Sequential(
            ConvBnRelu(enc_chs[4], d[0]),
            ConvBnRelu(d[0], d[0]),
        )

        self.block3 = DecoderBlock(d[0], enc_chs[3], d[1])  # + s4
        self.block2 = DecoderBlock(d[1], enc_chs[2], d[2])  # + s3
        self.block1 = DecoderBlock(d[2], enc_chs[1], d[3])  # + s2
        self.block0 = DecoderBlock(d[3], enc_chs[0], d[3] // 2)  # + s1

        self.head = nn.Sequential(
            nn.Conv2d(d[3] // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, features: EncoderFeatures, target_size: tuple[int, int]) -> torch.Tensor:
        x = self.bottleneck(features.s5)
        x = self.block3(x, features.s4)
        x = self.block2(x, features.s3)
        x = self.block1(x, features.s2)
        x = self.block0(x, features.s1)
        # Final upsample to original image size (e.g. 112 → 224)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return self.head(x)  # (B, 1, H, W) in [0, 1]
