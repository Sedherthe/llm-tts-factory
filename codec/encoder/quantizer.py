"""
Adapted from https://github.com/duchenzhuang/FSQ-pytorch/blob/main/quantizers/fsq.py#L41
"""

import torch
from torch import nn
from einops import rearrange


class FSQSTE(nn.Module):
    def __init__(self, levels):
        super().__init__()
        if levels:
            self.dim = len(levels)
            self._levels = torch.tensor(levels, dtype=torch.int32).view(1, 1, self.dim)
        else:
            self._levels = levels

        _levels = self._levels
        self.register_buffer("levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]),
                                    dim=0,
                                    dtype=torch.int32)
        self.register_buffer("_basis", _basis, persistent=False)

    def _scale_and_shift(self, zhat_normalized):
        half_width = self.levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self.levels // 2
        return (zhat - half_width) / half_width

    def indices_to_level_indices(self, indices):
        """ Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings """
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self.levels
        return codes_non_centered

    def to_codebook_index(self, zhat):
        """ Converts a `code` to an index in the codebook. """
        assert zhat.shape[-1] == self.dim
        zhat = self._scale_and_shift(zhat)
        indices = (zhat * self._basis).sum(dim = -1).round().to(torch.int32)
        return indices
    
    def from_codebook_index(self, indices):
        """ Inverse of `codes_to_indices`. """
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def forward(self, x):
        if self.levels is not None:

            half_levels = (self.levels - 1) * (1 - 1e-3) / 2
            offset = 0.5 - 0.5 * (self.levels % 2)
            shift = torch.tan(offset / half_levels)

            x = torch.tanh(x + shift) * half_levels - offset
            x = x + (x.round() - x).detach()
            x = x / (self.levels // 2)
        return x
