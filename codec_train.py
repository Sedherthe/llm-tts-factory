import os
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

from codec_model import FSQAutoEncoder
from codec_dataset import LJSpeechDataset
from codec.codec_decoder.decoder import SimpleDecoder

# -----------------------------------------------------------------------------
# Global Configuration
# -----------------------------------------------------------------------------
DATASET_ROOT = "/home/ubuntu/soma/data/lj_speech/LJSpeech-1.1"
SAMPLE_RATE = 32000

BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
NUM_WORKERS = 8

CKPT_DIR = "/home/ubuntu/soma/ckpt/suprano/suprano_codec/codec_v3"

# Freeze params
FREEZE_ENCODER = False
PRETRAINED_MODEL_PATH = "/home/ubuntu/soma/ckpt/suprano/suprano_codec/codec_1/step_42000.pt"

WANDB_PROJECT = "soprano-codec"
USE_WANDB = True
# -----------------------------------------------------------------------------


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


def save_mel_plot(original_mel, reconstructed_mel, original_title, reconstructed_title, save_path):
    """
    Helper function to plot and save Mel spectrograms.
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    axs[0].imshow(original_mel, aspect="auto", origin="lower")
    axs[0].set_title(original_title)

    axs[1].imshow(reconstructed_mel, aspect="auto", origin="lower")
    axs[1].set_title(reconstructed_title)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate(model, val_loader_it, val_loader, device, step, plot_dir, val_steps=10):
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for _ in range(val_steps):
            try:
                vdata = next(val_loader_it)
            except StopIteration:
                val_loader_it = iter(val_loader)
                vdata = next(val_loader_it)
            
            vaudio, _ = vdata
            vaudio = vaudio.squeeze().to(device)
            
            vmel_hat, vmel = model(vaudio)
            
            T_v = min(vmel_hat.shape[-1], vmel.shape[-1])
            vmel_hat = vmel_hat[..., :T_v]
            vmel = vmel[..., :T_v]
            
            val_loss += torch.mean(torch.abs(vmel_hat - vmel)).item()
    
    val_loss /= val_steps
    print(f"step {step} | val_loss {val_loss:.4f}")
    
    if USE_WANDB:
        wandb.log({"val/loss": val_loss}, step=step)

    # Val Plotting
    vmel_np = vmel[0].detach().cpu().numpy()
    vmel_hat_np = vmel_hat[0].detach().cpu().numpy()

    plot_path = os.path.join(plot_dir, f"val_step_{step:05d}.png")
    save_mel_plot(vmel_np, vmel_hat_np, "Val Original Mel", "Val Reconstructed Mel", plot_path)
    
    if USE_WANDB:
        wandb.log({"val/reconstruction": wandb.Image(plot_path)}, step=step)
    
    model.train()
    return val_loader_it


def main():
    if USE_WANDB:
        wandb.init(project=WANDB_PROJECT)

    # ------------------
    # Data Setup
    # ------------------
    dataset = LJSpeechDataset(
        root=DATASET_ROOT,
        sample_rate=SAMPLE_RATE,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=pad_collate
    )

    val_dataset = LJSpeechDataset(
        root=DATASET_ROOT,
        sample_rate=SAMPLE_RATE,
        mode='val'
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=pad_collate
    )
    val_loader_it = iter(val_loader)

    # ------------------
    # Model Config
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

    # Token rate sanity check
    total_hop = 2048 # This is the token hop rate. The mel hop rate is 512
    print(f"Token rate: {SAMPLE_RATE / total_hop:.2f} Hz")

    # ------------------
    # Model Setup
    # ------------------
    model = FSQAutoEncoder(encoder_cfg, decoder_cfg).to(device)

    # Here, we freeze the encoder and only train the decoder from scratch on learned encoder representation. 
    # This is to test whether the encoder has learned how to represent the audio well enough. 
    if FREEZE_ENCODER:
        if os.path.exists(PRETRAINED_MODEL_PATH):
            print(f"Loading model from {PRETRAINED_MODEL_PATH}")
            model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
        else:
            print(f"Warning: Pretrained model path {PRETRAINED_MODEL_PATH} not found.")

        # fix encoder; train only the decoder; reset the decoder weights.
        for param in model.encoder.parameters():
            param.requires_grad = False

        for name, p in model.named_parameters():
            if "quant" in name:
                print(name, p.requires_grad)

        model.decoder = SimpleDecoder(**decoder_cfg).to(device)

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    plot_dir = os.path.join(CKPT_DIR, "plots")
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    step = 0

    # ------------------
    # Training Loop
    # ------------------
    for epoch in range(NUM_EPOCHS):

        for epoch_step, data in tqdm(enumerate(loader), total=len(loader)):

            step += 1

            audio, lengths = data
            audio = audio.squeeze().to(device)

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
                if USE_WANDB:
                    wandb.log({"train/loss": loss.item()}, step=step)

            if step % 400 == 0:
                val_loader_it = evaluate(
                    model, val_loader_it, val_loader, device, step, plot_dir, 
                    val_steps=10
                )

                with torch.no_grad():
                    # FSQ bin usage
                    z = model.encoder.encode(mel)
                    indices = model.encoder.quant.to_codebook_index(z)
                    total_bins = int(torch.prod(torch.tensor(model.encoder.quant.levels)))
                    unique_bins = len(torch.unique(indices))
                    print(f"[FSQ] unique bins: {unique_bins} / {total_bins}. Lens: {indices.shape} {z.shape}")
                    
                    if USE_WANDB:
                        wandb.log({"train/unique_bins": unique_bins}, step=step)

            if step % 200 == 0:
                mel_np = mel[0].detach().cpu().numpy()
                mel_hat_np = mel_hat[0].detach().cpu().numpy()

                plot_path = os.path.join(plot_dir, f"step_{step:05d}.png")
                save_mel_plot(mel_np, mel_hat_np, "Original Mel", "Reconstructed Mel", plot_path)

                if USE_WANDB:
                    wandb.log({"train/reconstruction": wandb.Image(plot_path)}, step=step)

            if step % 1000 == 0:
                ckpt_path = os.path.join(CKPT_DIR, f"step_{step:05d}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
