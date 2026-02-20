"""
Adapted from https://github.com/gemelo-ai/vocos
"""
from typing import Optional

import torchaudio
import torch
from torch import nn

from .quantizer import FSQSTE

def safe_log(x: torch.Tensor, clip_val: float = 5e-3) -> torch.Tensor:
    return torch.log(torch.clip(x, min=clip_val))


class SimpleMLP(nn.Module):
    def __init__(self,
        dim,
        intermediate_dim,
    ):
        super().__init__()
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        dw_kernel_size: int = 7,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=dw_kernel_size, padding=dw_kernel_size//2, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = SimpleMLP(dim, intermediate_dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class VocosBackbone(nn.Module):
    """
    Vocos backbone module built with ConvNeXt blocks.

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling.
    """

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        input_kernel_size: int = 7,
        dw_kernel_size: int = 7,
        layer_scale_init_value: Optional[float] = None,
        pad: str = 'zeros',
    ):
        super().__init__()
        self.input_channels = input_channels
        self.dim = dim
        self.embed = nn.Conv1d(
            input_channels,
            dim,
            kernel_size=input_kernel_size,
            padding=input_kernel_size//2,
            padding_mode=pad
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.convnext = nn.ModuleList([
            ConvNeXtBlock(
                dim=dim,
                intermediate_dim=intermediate_dim,
                dw_kernel_size=dw_kernel_size,
                layer_scale_init_value=layer_scale_init_value or 1 / num_layers**0.5,
            )
            for _ in range(num_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, L), where B is the batch size,
                        C denotes output features, and L is the sequence length.

        Returns:
            Tensor: Output of shape (B, L, H), where B is the batch size, L is the sequence length,
                    and H denotes the model dimension.
        """
        x = self.embed(x) # (B, C, L)
        x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x)
        x = self.final_layer_norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        return x


class Encoder(nn.Module):
    def __init__(self,
        num_input_mels=50,
        mel_hop_length=512,
        mel_hop_scale=0.25,
        encoder_num_layers=8,
        encoder_dim=768,
        encoder_intermediate_dim=None,
        fsq_levels=[8, 8, 5, 5, 5],
        dw_kernel=5,
    ):
        super().__init__()
        self.downsample_scale = 2048 // mel_hop_length
        self.mel_hop_length = mel_hop_length
        self.mel_n_fft = int(mel_hop_length/mel_hop_scale)
        self.encoder_dim = encoder_dim
        self.encoder_intermediate_dim = encoder_intermediate_dim if encoder_intermediate_dim else encoder_dim*3
        self.encoder_num_layers = encoder_num_layers
        self.encoder_initial_channels = num_input_mels
        self.bottleneck_channels = 5

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=32000,
            n_fft=self.mel_n_fft,
            hop_length=self.mel_hop_length,
            n_mels=num_input_mels,
            center=True,
            power=1,
        )
        self.encoder = VocosBackbone(input_channels=self.encoder_initial_channels,
            dim=self.encoder_dim,
            intermediate_dim=self.encoder_intermediate_dim,
            num_layers=self.encoder_num_layers,
            input_kernel_size=1,
            dw_kernel_size=dw_kernel,
            pad='zeros'
        )
        self.downsampler = nn.Linear(self.encoder_dim, self.bottleneck_channels)
        self.quant = FSQSTE(levels=fsq_levels)

    def encode(self, x):
        x = self.encoder(x)

        # import pdb;pdb.set_trace()

        x = x[:, :, ::self.downsample_scale] # What the heck is this? Brute force downsampling from mel -> tokens. 
        x = x.transpose(1,2)
        x = self.downsampler(x)
        x = self.quant(x)
        return x

    def preprocess(self, audio):
        if audio.dim() == 2: # raw audio
            x = self.mel_spec(audio)
            x = safe_log(x)
        elif audio.dim() == 3: # mel spectrogram
            x = audio
        return x

    def forward(self, audio):
        x = self.preprocess(audio)

        # print("done preprocessing: ",x.shape)
        # import pdb;pdb.set_trace()

        x = self.encode(x)
        codes = self.quant.to_codebook_index(x)
        return codes
