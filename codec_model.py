import torch
from torch import nn
from codec.encoder.codec import Encoder
from codec.codec_decoder.decoder import SimpleDecoder


class FSQAutoEncoder(nn.Module):
    def __init__(self, encoder_cfg, decoder_cfg):
        super().__init__()
        self.encoder = Encoder(**encoder_cfg)
        self.decoder = SimpleDecoder(**decoder_cfg)

    def forward(self, audio):
        mel = self.encoder.preprocess(audio)
        z = self.encoder.encode(mel)      # FSQ STE output
        mel_hat = self.decoder(z)
        return mel_hat, mel
