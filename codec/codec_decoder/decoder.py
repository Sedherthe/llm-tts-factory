import torch
from torch import nn
from codec.encoder.codec import VocosBackbone


class SimpleDecoder(nn.Module):
    def __init__(
        self,
        n_mels=50,
        encoder_dim=768,
        bottleneck_channels=5,
        num_layers=8,   
        intermediate_dim=None,
        upsample_scale=4,
        dw_kernel=5,
    ):
        super().__init__()

        intermediate_dim = intermediate_dim or encoder_dim * 3
        self.upsample_scale = upsample_scale

        # project FSQ channels back to model dim
        self.in_proj = nn.Linear(bottleneck_channels, encoder_dim)

        # ConvNeXt backbone
        self.backbone = VocosBackbone(
            input_channels=encoder_dim,
            dim=encoder_dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            input_kernel_size=1,
            dw_kernel_size=dw_kernel,
        )

        # output mel projection
        self.out_proj = nn.Conv1d(encoder_dim, n_mels, kernel_size=1)

    def forward(self, z):
        """
        z: (B, T_latent, bottleneck_channels)
        """
        z = self.in_proj(z)              # (B, T_latent, D)
        z = z.transpose(1, 2)            # (B, D, T_latent)

        # naive upsampling (good enough for now)
        z = z.repeat_interleave(self.upsample_scale, dim=2)

        z = self.backbone(z)             # (B, D, T_mel)
        mel_hat = self.out_proj(z)       # (B, n_mels, T_mel)
        return mel_hat
