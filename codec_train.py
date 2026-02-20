import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import wandb
from tqdm import tqdm


from codec_model import FSQAutoEncoder
from codec_dataset import LJSpeechDataset
from codec.codec_decoder.decoder import SimpleDecoder


wandb.init(project="soprano-codec")


def pad_collate(batch):
    """
    batch: list of tensors [(1, T1), (1, T2), ...]
    """
    lengths = torch.tensor([x.shape[-1] for x in batch])

    max_len = lengths.max().item()

    padded = [
        F.pad(x, (0, max_len - x.shape[-1]))
        for x in batch
    ]

    audio = torch.stack(padded)  # (B, 1, T_max)
    return audio, lengths

def plot_mels():
    pass



dataset = LJSpeechDataset(
    root="/home/ubuntu/soma/data/lj_speech/LJSpeech-1.1",
    sample_rate=32000,
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
    collate_fn=pad_collate
)

val_dataset = LJSpeechDataset(
    root="/home/ubuntu/soma/data/lj_speech/LJSpeech-1.1",
    sample_rate=32000,
    mode='val'
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
    collate_fn=pad_collate
)
val_loader_it = iter(val_loader)

# ------------------
# Config
# ------------------

encoder_cfg = dict(
    num_input_mels=50,
    mel_hop_length=512,
    encoder_dim=768,
    encoder_num_layers=8,
    fsq_levels=[8, 8, 5, 5, 5],
)

decoder_cfg = dict(
    n_mels=50,
    encoder_dim=768,
    bottleneck_channels=5,
    num_layers=8,
    upsample_scale=2048 // 512,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 3. Token rate sanity check
# downsample_scale = 2048 // mel_hop_length
# total_hop = mel_hop_length * downsample_scale = 2048
# rate = sr / 2048
sr = 32000
total_hop = 2048 # This is the token hop rate. The mel hop rate is 512
print(f"Token rate: {sr / total_hop:.2f} Hz")


# ------------------
# Setup
# ------------------

model = FSQAutoEncoder(encoder_cfg, decoder_cfg).to(device)

freeze_encoder = False
if freeze_encoder:
    model_ckpt_path = "/home/ubuntu/soma/ckpt/suprano/suprano_codec/codec_1/step_42000.pt"

    if os.path.exists(model_ckpt_path):
        print(f"Loading model from {model_ckpt_path}")
        model.load_state_dict(torch.load(model_ckpt_path))

    # fix encoder. train only the decoder. reset the decoder weights.
    for param in model.encoder.parameters():
        param.requires_grad = False

    for name, p in model.named_parameters():
        if "quant" in name:
            print(name, p.requires_grad)

    model.decoder = SimpleDecoder(**decoder_cfg).to(device)

optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

import pdb;pdb.set_trace()


root_dir = "/home/ubuntu/soma/ckpt/suprano/suprano_codec"
ckpt_dir = os.path.join(root_dir, "codec_v2")
plot_dir = os.path.join(ckpt_dir, "plots")
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

num_epochs = 100
step = 0

for epoch in range(num_epochs):

    for epoch_step, data in tqdm(enumerate(loader), total=len(loader)):

        step += 1

        audio, lengths = data
        audio = audio.squeeze().to(device)

        # import pdb;pdb.set_trace()

        mel_hat, mel = model(audio)

        # crop to match length (upsampling can overshoot)
        T = min(mel_hat.shape[-1], mel.shape[-1])
        mel_hat = mel_hat[..., :T]
        mel = mel[..., :T]

        loss = torch.mean(torch.abs(mel_hat - mel))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"step {step} | loss {loss.item():.4f}")
            wandb.log({"train/loss": loss.item()}, step=step)

        if step % 400 == 0:
            val_loss = 0.0
            val_steps = 10
            model.eval()
            with torch.no_grad():
                for _ in range(val_steps):
                    try:
                        vdata = next(val_loader_it)
                    except StopIteration:
                        val_loader_it = iter(val_loader)
                        vdata = next(val_loader_it)
                    
                    vaudio, vlengths = vdata
                    vaudio = vaudio.squeeze().to(device)
                    
                    vmel_hat, vmel = model(vaudio)
                    
                    T_v = min(vmel_hat.shape[-1], vmel.shape[-1])
                    vmel_hat = vmel_hat[..., :T_v]
                    vmel = vmel[..., :T_v]
                    
                    val_loss += torch.mean(torch.abs(vmel_hat - vmel)).item()
            
            val_loss /= val_steps
            print(f"step {step} | val_loss {val_loss:.4f}")
            wandb.log({"val/loss": val_loss}, step=step)

            # Val Plotting
            vmel_np = vmel[0].detach().cpu().numpy()
            vmel_hat_np = vmel_hat[0].detach().cpu().numpy()

            fig, axs = plt.subplots(2, 1, figsize=(10, 6))

            axs[0].imshow(vmel_np, aspect="auto", origin="lower")
            axs[0].set_title("Val Original Mel")

            axs[1].imshow(vmel_hat_np, aspect="auto", origin="lower")
            axs[1].set_title("Val Reconstructed Mel")

            plt.tight_layout()
            plt.savefig(f"{plot_dir}/val_step_{step:05d}.png")
            wandb.log({"val/reconstruction": wandb.Image(f"{plot_dir}/val_step_{step:05d}.png")}, step=step)
            plt.close()
            
            model.train()

        if step % 400 == 0:
            with torch.no_grad():
                # 1. FSQ bin usage
                z = model.encoder.encode(mel)
                indices = model.encoder.quant.to_codebook_index(z)
                total_bins = int(torch.prod(model.encoder.quant.levels))
                unique_bins = len(torch.unique(indices))
                print(f"[FSQ] unique bins: {unique_bins} / {total_bins}. Lens: {indices.shape} {z.shape}")
                wandb.log({"train/unique_bins": unique_bins}, step=step)

        if step % 200 == 0:
            mel_np = mel[0].detach().cpu().numpy()
            mel_hat_np = mel_hat[0].detach().cpu().numpy()

            fig, axs = plt.subplots(2, 1, figsize=(10, 6))

            axs[0].imshow(mel_np, aspect="auto", origin="lower")
            axs[0].set_title("Original Mel")

            axs[1].imshow(mel_hat_np, aspect="auto", origin="lower")
            axs[1].set_title("Reconstructed Mel")

            plt.tight_layout()
            plt.savefig(f"{plot_dir}/step_{step:05d}.png")
            wandb.log({"train/reconstruction": wandb.Image(f"{plot_dir}/step_{step:05d}.png")}, step=step)
            plt.close()


        if step % 1000 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"step_{step:05d}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
