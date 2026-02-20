import torch
import torch.nn.functional as F
import torchaudio

def dynamic_range_compression_torch(x, C=1, clip_val: float = 5e-3):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

# Mel Spectrogram initialization logic
# Note: Should we initialize the transform here or pass it in?
# For simplicity, we can provide a factory function or a class.
# But original code used global `mel_transform`.
# To keep it clean, let's wrap it in a class or function that returns the transform.

class MelSpectrogramWrapper(torch.nn.Module):
    def __init__(self, sample_rate=32000, n_fft=2048, hop_length=512, n_mels=50):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, 
            n_mels=n_mels, center=True, power=1
        )
    
    def forward(self, audio):
        mel_spec = self.mel_transform(audio)
        mel_spec = spectral_normalize_torch(mel_spec)
        return mel_spec

def feature_matching_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses

def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

# -----------------
# Multi-Resolution STFT Loss
# -----------------

def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and return linear mag spectogram."""
    # x: (B, T)
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window, center=True, return_complex=True)
    x_mag = torch.abs(x_stft)
    return x_mag

def spectral_convergence_loss(x_mag, y_mag):
    """
    Spectral convergence loss.
    """
    return torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + 1e-7)

def log_magnitude_loss(x_mag, y_mag):
    """
    Log-magnitude L1 loss.
    """
    return F.l1_loss(torch.log(x_mag), torch.log(y_mag))

class STFTLoss(torch.nn.Module):
    """
    STFT Loss module.
    """
    def __init__(self, fft_size, hop_size, win_length, window="hann_window"):
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))

    def forward(self, x, y):
        """
        Args:
            x (Tensor): Predicted audio (B, T).
            y (Tensor): Target audio (B, T).
        """
        x_mag = stft(x, self.fft_size, self.hop_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.hop_size, self.win_length, self.window)
        
        # Add epsilon to prevent log(0)
        x_mag = torch.clamp(x_mag, min=1e-7)
        y_mag = torch.clamp(y_mag, min=1e-7)
        
        sc_loss = spectral_convergence_loss(x_mag, y_mag)
        mag_loss = log_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss

class MultiResolutionSTFTLoss(torch.nn.Module):
    """
    Multi-Resolution STFT Loss module.
    """
    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240], # Standard HiFiGAN config
                 window="hann_window", factor_sc=1.0, factor_mag=1.0):
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        
        self.stft_losses = torch.nn.ModuleList()
        for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses.append(STFTLoss(fs, hs, wl, window))
            
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self, x, y):
        sc_loss = 0.0
        mag_loss = 0.0
        
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss
