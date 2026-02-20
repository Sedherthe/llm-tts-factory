"""
Training script for Soprano Decoder (Vocos).
Freezes LLM and trains Decoder with GAN loss.
"""
import argparse
import pathlib
import random
import time
import os
import wandb
import matplotlib.pyplot as plt
import io

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub import hf_hub_download

from dataset_e2e import AudioDataset, SAMPLES_PER_TOKEN
from decoder.decoder import SopranoDecoder
from decoder.discriminator import Discriminator
from decoder.losses import MelSpectrogramWrapper, feature_matching_loss, discriminator_loss, generator_loss, MultiResolutionSTFTLoss

# Initialize Mel Spectrogram Wrapper
mel_fn = MelSpectrogramWrapper().to('cuda')



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",
        required=False,
        default="./example_dataset",
        type=pathlib.Path
    )
    parser.add_argument("--save-dir",
        required=True,
        type=pathlib.Path
    )
    parser.add_argument("--use-disc",
        action="store_true",
        help="Whether to use discriminator and GAN losses. Defaults to False (only reconstruction loss)."
    )
    return parser.parse_args()

args = get_args()

# training hyperparameters
device = 'cuda:0'
seed = 1337
max_lr = 2e-4 # Lower LR for GAN usually
warmup_ratio = 0.2
cooldown_ratio = 0.1
min_lr = 0.1 * max_lr
batch_size = 64 # 64
grad_accum_steps = 1
seq_len = 1024
SEGMENT_SIZE_SAMPLES = 32768 # ~1 sec (16 tokens)
val_freq = 250
text_factor = 0.0 
max_steps = 200000
betas = (0.8, 0.99) # GAN betas
weight_decay = 0.1
train_dataset_path = f'{args.input_dir}/train.json'
val_dataset_path = f'{args.input_dir}/val.json'
save_path = args.save_dir
os.makedirs(save_path, exist_ok=True)

# Loss Weights
lambda_mel = 45.0
lambda_fm = 2.0
lambda_gen = 1.0
lambda_stft = 1.0

def worker_seed_init(_):
    worker_seed = torch.initial_seed() % (2**32-1)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_lr(it): # WSD schedule
    if it<warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it<max_steps-cooldown_steps:
        return max_lr
    return min_lr + (max_lr-min_lr) * ((max_steps-it) / cooldown_steps)

# ... (Previous code)

# Initialize Mel Spectrogram Wrapper
# Place this after device definition (Line 66)
# But here we are replacing lines 26-43 and 102-130.
# I will do this in chunks.


def collate_pack(batch_in):
    # batch_in is list of (text, wav)
    texts = [x[0] for x in batch_in]
    wavs = [x[1] for x in batch_in]
    aud_token_lens = [x[2] for x in batch_in]
    
    tokens_batch = tokenizer(texts, padding=True, return_tensors='pt')
    input_ids = tokens_batch['input_ids'] # (B, T)
    
    # We need to process each sample to align audio
    # Since lengths vary, we process list then pad
    
    batch_tokens_list = []
    batch_audio_list = []
    
    for i in range(len(texts)):
        # Get raw tokens without padding for alignment logic
        raw_tokens = tokenizer(texts[i], padding=False, truncation=False)['input_ids']
        tokens = torch.tensor(raw_tokens, dtype=torch.long) # remove last token as per original logic? Or keep?
        
        # Taking raw tokens
        tokens = torch.tensor(raw_tokens, dtype=torch.long)
        
        wav = wavs[i]
        num_aud_tokens = aud_token_lens[i]
        aligned_audio = torch.zeros(num_aud_tokens * SAMPLES_PER_TOKEN, dtype=torch.float32)
        wav_ptr = 0
        is_audio = (tokens > 3) & (tokens <= 8003)
        audio_indices = torch.where(is_audio)[0]
        
        if len(audio_indices) != num_aud_tokens:
            print(f"Audio token count mismatch: {len(audio_indices)} vs {num_aud_tokens}")
            print(texts[i])
            print(raw_tokens)
            print(is_audio)
            print(audio_indices)
        assert len(audio_indices) == num_aud_tokens, f"Audio token count mismatch: {len(audio_indices)} vs {num_aud_tokens}"

        for pos, idx in enumerate(audio_indices):
            if wav_ptr + SAMPLES_PER_TOKEN <= wav.size(0):
                aligned_audio[pos*SAMPLES_PER_TOKEN : (pos+1)*SAMPLES_PER_TOKEN] = wav[wav_ptr : wav_ptr+SAMPLES_PER_TOKEN]
                wav_ptr += SAMPLES_PER_TOKEN
            else:
                break
        
        batch_tokens_list.append(tokens)
        batch_audio_list.append(aligned_audio)

    # Pad Tokens
    # We need to return x and y. So we need sequence length T.
    # tokens are length T+1 (input + target).
    # Pad to max length
    batch_tokens = torch.nn.utils.rnn.pad_sequence(batch_tokens_list, batch_first=True, padding_value=0)
    
    # Pad Audio
    # Audio length = tokens_length * SAMPLES_PER_TOKEN
    # Since 0 token -> 2048 zeros, padding matches.
    batch_audio = torch.nn.utils.rnn.pad_sequence(batch_audio_list, batch_first=True, padding_value=0.0)
    
    x = batch_tokens[:, :-1]
    y = batch_tokens[:, 1:]
    
    # Audio matches x (length T)
    # batch_audio was created for length T+1 (full tokens).
    # We need audio corresponding to x.
    # Dimensions: batch_audio is (B, (T+1)*2048)
    # We want up to T*2048
    
    # Calculate max seq len of x
    max_len_x = x.size(1)
    gt_audio = batch_audio[:, :max_len_x * SAMPLES_PER_TOKEN]

    # Create Attention Mask
    # Start with x mask
    # x_mask = (x != 0) # Assumes 0 is padding value used above
    
    # Create Audio Mask (True where token is audio)
    audio_mask = (y > 3) & (y <= 8003) # Compute output based on y. I should start collecting outputs from the start_token in x.

    return x, y, gt_audio, audio_mask

tokenizer = AutoTokenizer.from_pretrained('ekwek/Soprano-80M')
tokenizer.padding_side = 'right' # Essential for training!

if __name__ == '__main__':
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_float32_matmul_precision('high')
    print(f"Save Path: {save_path}")

    # lr schedule
    warmup_steps = int(max_steps * warmup_ratio)
    cooldown_steps = int(max_steps * cooldown_ratio)

    # Initialize WandB
    wandb.init(project="soprano-decoder-only", config=vars(args))

    # 1. Load LLM and Freeze
    # Load custom trained checkpoint
    print("Loading LLM from custom checkpoint...")
    config = AutoConfig.from_pretrained('ekwek/Soprano-80M')
    model = AutoModelForCausalLM.from_config(config)

    from safetensors.torch import load_file
    # ckpt_path = "/home/ubuntu/soma/ckpt/suprano/suprano_llm/codec_1/model.safetensors"
    # ckpt_path = "/home/ubuntu/soma/ckpt/suprano/suprano_llm/codec_1_2/checkpoint-10000/model.safetensors"
    # ckpt_path = "/home/ubuntu/soma/ckpt/suprano/suprano_llm/codec_v2/v2/model.safetensors"
    ckpt_path = "/home/ubuntu/soma/ckpt/suprano/suprano_llm/codec_v2/v2/checkpoint-40000/model.safetensors"
    # ckpt_path = "/home/ubuntu/soma/ckpt/suprano/suprano_llm/codec_v2/v2/checkpoint-135000/model.safetensors"
    state_dict = load_file(ckpt_path)
    model.load_state_dict(state_dict)

    model.to(torch.bfloat16).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print("LLM Frozen.")
    
    # 2. Decoder
    print("Loading Decoder...")
    decoder = SopranoDecoder()
    # decoder_ckpt_path = "/home/ubuntu/soma/ckpt/suprano/suprano_deocoder/codec_1_fix_lens/decoder_step_10000.pth"
    # decoder_ckpt_path = "/home/ubuntu/soma/ckpt/suprano/suprano_deocoder/codec_v2/v2_40k_w_stft_loss/decoder_trained.pth"
    decoder_ckpt_path = "/home/ubuntu/soma/ckpt/suprano/suprano_deocoder/codec_v2/v2_40k_w_stft_loss_w_disc/decoder_step_141000.pth"
    decoder.load_state_dict(torch.load(decoder_ckpt_path, map_location='cpu'))
    
    decoder.to(device)
    decoder.train() 
    print("Decoder loaded.")

    # Initialize MR-STFT Loss
    mr_stft = MultiResolutionSTFTLoss().to(device)

    # 3. Discriminator
    discriminator = None
    if args.use_disc:
        print("Initializing Discriminator...")

        discriminator = Discriminator()

        disc_model_path = "/home/ubuntu/soma/ckpt/suprano/suprano_deocoder/codec_v2/v2_40k_w_stft_loss_w_disc/discriminator_step_141000.pth"
        discriminator.load_state_dict(torch.load(disc_model_path, map_location='cpu'))
        
        discriminator.to(device)
        discriminator.train()
    else:
        print("Training WITHOUT Discriminator (Reconstruction only).")

    # 4. Dataset
    dataset = AudioDataset(train_dataset_path)
    dataloader = DataLoader(dataset,
        batch_size=8, # change this back
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_seed_init,
        collate_fn=collate_pack,
    )
    dataloader_it = iter(dataloader)

    val_dataset = AudioDataset(val_dataset_path)
    val_dataloader = DataLoader(val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_seed_init,
        collate_fn=collate_pack,
    )
    val_dataloader_it = iter(val_dataloader)

    # import pdb;pdb.set_trace()

    # 5. Optimizers
    opt_g = torch.optim.AdamW(decoder.parameters(), max_lr, betas=betas, weight_decay=weight_decay)
    opt_d = None
    if args.use_disc:
        opt_d = torch.optim.AdamW(discriminator.parameters(), max_lr, betas=betas, weight_decay=weight_decay)

    start_step = 141001
    pbar = tqdm(range(start_step, max_steps), ncols=200, dynamic_ncols=True)
    for step in pbar:
        start = time.time()
        
        # Get Data
        try:
            batch_data = next(dataloader_it)
            if batch_data[0] is None: 
                dataloader_it = iter(dataloader)
                batch_data = next(dataloader_it)
            x, y, gt_audio, audio_mask = batch_data # Audio mask is the audio token mask for token prediction loss here. 

            # import pdb;pdb.set_trace()

        except StopIteration:
            dataloader_it = iter(dataloader)
            batch_data = next(dataloader_it)
            x, y, gt_audio, audio_mask = batch_data
            
        x, y = x.to(device), y.to(device)
        gt_audio = gt_audio.to(device) # (B, T_audio_samples)
        audio_mask = audio_mask.to(device)

        # Forward LLM (No Grad)
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                outputs = model(x, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1] # (B, T_total, D)
                hidden_states = hidden_states.to(torch.float32)

        # GATHER AUDIO LATENTS Logic
        # We need to extract only the hidden states where audio_mask is True.
        # Since number of audio tokens varies per sample, we gather and then Pad.
        
        gathered_states_list = []
        for b_idx in range(hidden_states.size(0)):
            mask = audio_mask[b_idx]
            valid_states = hidden_states[b_idx][mask]
            gathered_states_list.append(valid_states)

        decoder_in_padded = torch.nn.utils.rnn.pad_sequence(gathered_states_list, batch_first=True, padding_value=0.0)
        

        bsz = decoder_in_padded.size(0)
        max_aud_len = decoder_in_padded.size(1)
        audio_loss_mask = torch.zeros((bsz, max_aud_len), dtype=torch.bool, device=device)
        for b_idx in range(bsz):
            length = gathered_states_list[b_idx].size(0)
            audio_loss_mask[b_idx, :length] = True

        # ---------------------
        # Train Discriminator
        # ---------------------
        d_loss_item = 0.0
        if args.use_disc:
            opt_d.zero_grad()
            
            # Generator Forward (Detach for D training)
            decoder_in = decoder_in_padded.transpose(1, 2) # (B, C, T)
            fake_audio = decoder(decoder_in) # (B, 1, T_audio_gen)
            if fake_audio.size(1) == 1: fake_audio = fake_audio.squeeze(1)
            
        min_len = min(fake_audio.size(1), gt_audio.size(1))
        fake_audio = fake_audio[:, :min_len]
        real_audio = gt_audio[:, :min_len]

        # ---------------------
        # Train Discriminator (on Crops)
        # ---------------------
        d_loss_item = 0.0
        if args.use_disc:
            opt_d.zero_grad()
            
            # --- Random Cropping Logic ---
            real_crop_list = []
            fake_crop_list = []
            
            for b_idx in range(bsz):
                # Calculate valid audio length for this sample
                valid_len = gathered_states_list[b_idx].size(0) * SAMPLES_PER_TOKEN
                valid_len = min(valid_len, min_len) # Clamp to generated length
                
                if valid_len <= SEGMENT_SIZE_SAMPLES:
                    # Pad if shorter
                    pad_len = SEGMENT_SIZE_SAMPLES - valid_len
                    r_c = torch.nn.functional.pad(real_audio[b_idx, :valid_len], (0, pad_len))
                    f_c = torch.nn.functional.pad(fake_audio[b_idx, :valid_len], (0, pad_len))
                else:
                    # Random Crop
                    start_idx = random.randint(0, valid_len - SEGMENT_SIZE_SAMPLES)
                    r_c = real_audio[b_idx, start_idx : start_idx + SEGMENT_SIZE_SAMPLES]
                    f_c = fake_audio[b_idx, start_idx : start_idx + SEGMENT_SIZE_SAMPLES]
                
                real_crop_list.append(r_c)
                fake_crop_list.append(f_c)
            
            real_crops = torch.stack(real_crop_list).unsqueeze(1) # (B, 1, T_seg)
            fake_crops = torch.stack(fake_crop_list).unsqueeze(1).detach() # Detach for D update

            # Disc Forward
            y_d_rs, y_d_gs, _, _ = discriminator(real_crops, fake_crops)
            d_loss, _, _ = discriminator_loss(y_d_rs, y_d_gs)
            
            d_loss.backward()
            norm_d = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            opt_d.step()
            d_loss_item = d_loss.item()
        
        # ---------------------
        # Train Generator
        # ---------------------
        opt_g.zero_grad()
        
        # Re-run generator (or reuse graph if not detached incorrectly)
        decoder_in = decoder_in_padded.transpose(1, 2)
        fake_audio = decoder(decoder_in)
        if fake_audio.size(1) == 1: fake_audio = fake_audio.squeeze(1)
        
        min_len = min(fake_audio.size(1), gt_audio.size(1))
        fake_audio = fake_audio[:, :min_len]
        real_audio = gt_audio[:, :min_len]
        
        # Re-Crop for Generator (Same logic, new random crop or reuse?)
        # Ideally reuse same crops if we didn't update D? But we did.
        # Or standard GAN: Update D, then update G.
        # Usually we crop again (stochasticity helps).
        # We need "fake_crops_g" (with grad).
        
        real_crop_list_g = []
        fake_crop_list_g = []
        if args.use_disc:
             for b_idx in range(bsz):
                valid_len = gathered_states_list[b_idx].size(0) * SAMPLES_PER_TOKEN
                valid_len = min(valid_len, min_len)
                
                if valid_len <= SEGMENT_SIZE_SAMPLES:
                    pad_len = SEGMENT_SIZE_SAMPLES - valid_len
                    r_c = torch.nn.functional.pad(real_audio[b_idx, :valid_len], (0, pad_len))
                    f_c = torch.nn.functional.pad(fake_audio[b_idx, :valid_len], (0, pad_len))
                else:
                    start_idx = random.randint(0, valid_len - SEGMENT_SIZE_SAMPLES)
                    r_c = real_audio[b_idx, start_idx : start_idx + SEGMENT_SIZE_SAMPLES]
                    f_c = fake_audio[b_idx, start_idx : start_idx + SEGMENT_SIZE_SAMPLES]
                
                real_crop_list_g.append(r_c)
                fake_crop_list_g.append(f_c)
            
             real_crops_g = torch.stack(real_crop_list_g).unsqueeze(1)
             fake_crops_g = torch.stack(fake_crop_list_g).unsqueeze(1)

        # Losses
        # Create Mel Mask
        # We need mask corresponding to decoder_in (Audio Only)
        # 1 Token = SAMPLES_PER_TOKEN audio samples
        # Mel Hop Length = 512
        # Ratio = SAMPLES_PER_TOKEN / 512 = 4 frames per token
        frames_per_token = SAMPLES_PER_TOKEN // 512
        
        # Expand audio_loss_mask: (B, T_aud) -> (B, T_aud*4)
        mel_mask = audio_loss_mask.repeat_interleave(frames_per_token, dim=1)
        
        pred_mel = mel_fn(fake_audio)
        gt_mel = mel_fn(real_audio)
        
        # Crop to min length
        min_mel_len = min(pred_mel.size(2), gt_mel.size(2), mel_mask.size(1))
        pred_mel = pred_mel[:, :, :min_mel_len]
        gt_mel = gt_mel[:, :, :min_mel_len]
        mel_mask = mel_mask[:, :min_mel_len]
        
        loss_mel_raw = torch.nn.functional.l1_loss(pred_mel, gt_mel, reduction='none')
        # Apply mask
        loss_mel = (loss_mel_raw * mel_mask.unsqueeze(1)).sum() / (mel_mask.sum() * pred_mel.size(1) + 1e-6)
        
        # Multi-Resolution STFT Loss
        # Masking: Apply sample_mask to inputs
        # Create sample_mask
        sample_mask = audio_loss_mask.repeat_interleave(SAMPLES_PER_TOKEN, dim=1)
        # Crop to min_len
        sample_mask = sample_mask[:, :min_len]
        
        # Apply mask (Assuming padding is zeros, but ensuring it)
        # This prevents the model from being penalized for non-zero output in padding region (if target is zero)
        # However, standard MR-STFT computes loss on the full spectrogram.
        # If we zeroes out padding, STFT of zero is zero. Loss over zero-zero region is zero.
        # But we divide by total elements (implicitly in mean).
        # MR-STFT implementation uses averaging over batch/time. 
        # Ideally we should use a masked reduction.
        # But for now, let's just zero out inputs.
        
        sc_loss, mag_loss = mr_stft(fake_audio * sample_mask, real_audio * sample_mask)
        
        loss_fm = torch.tensor(0.0, device=device)
        loss_gen = torch.tensor(0.0, device=device)
        
        
        if args.use_disc:
            # Disc Forward Again (No Detach, using Crops)
            y_d_rs, y_d_gs, fmap_rs, fmap_gs = discriminator(real_crops_g, fake_crops_g)
            loss_fm = feature_matching_loss(fmap_rs, fmap_gs)
            loss_gen, _ = generator_loss(y_d_gs)
        
        total_loss_g = (lambda_mel * loss_mel) + (lambda_gen * loss_gen) + (lambda_fm * loss_fm) + (lambda_stft * (sc_loss + mag_loss))
        
        total_loss_g.backward()
        norm_g = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        
        # LR Update
        lr = get_lr(step)
        for param_group in opt_g.param_groups: param_group['lr'] = lr
        if args.use_disc:
            for param_group in opt_d.param_groups: param_group['lr'] = lr / 2
            
        opt_g.step()

        end = time.time()
        dt = (end-start)*1000
        
        tqdm_log = f'mel: {loss_mel.item():.3f} | gen: {loss_gen.item():.3f} | sc: {sc_loss.item():.3f} | mag: {mag_loss.item():.3f} | fm: {loss_fm.item():.3f} | d: {d_loss_item:.3f} | lr: {lr:.2e} | time: {dt:.2f} ms'
        pbar.set_description(tqdm_log)

        # WandB Logging
        log_dict = {
            "train/loss_mel": loss_mel.item(),
            "train/loss_gen": loss_gen.item(),
            "train/loss_fm": loss_fm.item(),
            "train/loss_d": d_loss_item,
            "train/lr": lr,
            "train/total_loss_g": total_loss_g.item(),
            "train/loss_sc": sc_loss.item(),
            "train/loss_mag": mag_loss.item()
        }
        
        # Validation Loop & Logging
        if step % val_freq == 0:
            decoder.eval()
            if discriminator: discriminator.eval()
            
            val_mel_loss_accum = 0.0
            val_gen_loss_accum = 0.0
            val_fm_loss_accum = 0.0
            val_d_loss_accum = 0.0
            val_sc_loss_accum = 0.0
            val_mag_loss_accum = 0.0
            val_steps = 10 # Check 10 batches for speed
            
            with torch.no_grad():
                for _ in range(val_steps):
                    try:
                        val_batch = next(val_dataloader_it)
                    except StopIteration:
                        val_dataloader_it = iter(val_dataloader)
                        val_batch = next(val_dataloader_it)
                        
                    vx, vy, vgt_audio, vaudio_mask = val_batch
                    vx, vy = vx.to(device), vy.to(device)
                    vgt_audio = vgt_audio.to(device)
                    vaudio_mask = vaudio_mask.to(device)
                    
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        voutputs = model(vx, output_hidden_states=True)
                        v_hidden = voutputs.hidden_states[-1].to(torch.float32)
                    
                    # GATHER AUDIO LATENTS Logic
                    v_gathered_states_list = []
                    for b_idx in range(v_hidden.size(0)):
                        mask = vaudio_mask[b_idx]
                        v_valid_states = v_hidden[b_idx][mask]
                        v_gathered_states_list.append(v_valid_states)
                    
                    v_in_padded = torch.nn.utils.rnn.pad_sequence(v_gathered_states_list, batch_first=True, padding_value=0.0)
                    
                    # Calculate Audio-specific Mask for Loss
                    v_bsz = v_in_padded.size(0)
                    v_max_aud_len = v_in_padded.size(1)
                    v_audio_loss_mask = torch.zeros((v_bsz, v_max_aud_len), dtype=torch.bool, device=device)
                    for b_idx in range(v_bsz):
                        length = v_gathered_states_list[b_idx].size(0)
                        v_audio_loss_mask[b_idx, :length] = True

                    v_in = v_in_padded.transpose(1, 2)
                    v_fake_audio = decoder(v_in)
                    if v_fake_audio.size(1) == 1: v_fake_audio = v_fake_audio.squeeze(1)
                    
                    min_len_v = min(v_fake_audio.size(1), vgt_audio.size(1))
                    v_fake_audio = v_fake_audio[:, :min_len_v]
                    v_real_audio = vgt_audio[:, :min_len_v]
                    
                    # Val Mel Loss
                    frames_per_token_v = SAMPLES_PER_TOKEN // 512
                    v_mel_mask = v_audio_loss_mask.repeat_interleave(frames_per_token_v, dim=1)
                    
                    v_pred_mel = mel_fn(v_fake_audio)
                    v_gt_mel = mel_fn(v_real_audio)
                    
                    min_mel_len_v = min(v_pred_mel.size(2), v_gt_mel.size(2), v_mel_mask.size(1))
                    v_pred_mel = v_pred_mel[:, :, :min_mel_len_v]
                    v_gt_mel = v_gt_mel[:, :, :min_mel_len_v]
                    v_mel_mask = v_mel_mask[:, :min_mel_len_v]
                    
                    v_mel_loss_raw = torch.nn.functional.l1_loss(v_pred_mel, v_gt_mel, reduction='none')
                    v_mel_loss = (v_mel_loss_raw * v_mel_mask.unsqueeze(1)).sum() / (v_mel_mask.sum() * v_pred_mel.size(1) + 1e-6)
                    
                    val_mel_loss_accum += v_mel_loss.item()
                    
                    # Val MR-STFT
                    v_sample_mask = v_audio_loss_mask.repeat_interleave(SAMPLES_PER_TOKEN, dim=1)[:, :min_len_v]
                    v_sc_loss, v_mag_loss = mr_stft(v_fake_audio * v_sample_mask, v_real_audio * v_sample_mask)
                    val_sc_loss_accum += v_sc_loss.item()
                    val_mag_loss_accum += v_mag_loss.item()
                    
                    if args.use_disc:
                         # Disc Forward with Cropping (Similar to Training)
                        v_real_crop_list = []
                        v_fake_crop_list = []
                        
                        v_min_len = min(v_fake_audio.size(1), v_real_audio.size(1))
                        
                        for b_idx in range(v_bsz):
                            v_valid_len = v_gathered_states_list[b_idx].size(0) * SAMPLES_PER_TOKEN
                            v_valid_len = min(v_valid_len, v_min_len)
                            
                            if v_valid_len <= SEGMENT_SIZE_SAMPLES:
                                v_pad_len = SEGMENT_SIZE_SAMPLES - v_valid_len
                                vr_c = torch.nn.functional.pad(v_real_audio[b_idx, :v_valid_len], (0, v_pad_len))
                                vf_c = torch.nn.functional.pad(v_fake_audio[b_idx, :v_valid_len], (0, v_pad_len))
                            else:
                                # Random Crop (or fixed center crop for determinism in val?)
                                # Random is fine as it averages out over batches/epochs.
                                v_start_idx = random.randint(0, v_valid_len - SEGMENT_SIZE_SAMPLES)
                                vr_c = v_real_audio[b_idx, v_start_idx : v_start_idx + SEGMENT_SIZE_SAMPLES]
                                vf_c = v_fake_audio[b_idx, v_start_idx : v_start_idx + SEGMENT_SIZE_SAMPLES]
                            
                            v_real_crop_list.append(vr_c)
                            v_fake_crop_list.append(vf_c)
                        
                        v_real_crops = torch.stack(v_real_crop_list).unsqueeze(1)
                        v_fake_crops = torch.stack(v_fake_crop_list).unsqueeze(1)

                        vy_d_rs, vy_d_gs, vfmap_rs, vfmap_gs = discriminator(v_real_crops, v_fake_crops)
                        v_fm_loss = feature_matching_loss(vfmap_rs, vfmap_gs)
                        v_gen_loss, _ = generator_loss(vy_d_gs)
                        v_d_loss, _, _ = discriminator_loss(vy_d_rs, vy_d_gs)
                        
                        val_gen_loss_accum += v_gen_loss.item()
                        val_fm_loss_accum += v_fm_loss.item()
                        val_d_loss_accum += v_d_loss.item()

            # Average metrics
            val_log = {
                "val/loss_mel": val_mel_loss_accum / val_steps,
                "val/loss_gen": val_gen_loss_accum / val_steps,
                "val/loss_fm": val_fm_loss_accum / val_steps,
                "val/loss_d": val_d_loss_accum / val_steps,
                "val/loss_sc": val_sc_loss_accum / val_steps,
                "val/loss_mag": val_mag_loss_accum / val_steps
            }
            log_dict.update(val_log)

            # Generate Mel Images (from last val batch)
            gen_mel = mel_fn(v_fake_audio[0:1]).squeeze(0).cpu().numpy()
            gt_mel = mel_fn(v_real_audio[0:1]).squeeze(0).cpu().numpy()
            
            # Create Plot
            fig, ax = plt.subplots(2, 1, figsize=(10, 6))
            ax[0].imshow(gt_mel, aspect='auto', origin='lower')
            ax[0].set_title("Ground Truth Mel")
            ax[1].imshow(gen_mel, aspect='auto', origin='lower')
            ax[1].set_title("Generated Mel (Val)")
            plt.tight_layout()
            
            # Log to WandB
            log_dict["val/mel_spectrograms"] = wandb.Image(fig)
            plt.close(fig)
            
            # Return to train mode
            decoder.train()
            if discriminator: discriminator.train()

        # Save Checkpoint
        if step > 0 and step % 3000 == 0:
            print(f"Saving checkpoint at step {step}...")
            ckpt_name_dec = f"decoder_step_{step}.pth"
            ckpt_name_disc = f"discriminator_step_{step}.pth"
            torch.save(decoder.state_dict(), save_path / ckpt_name_dec)
            if discriminator:
                torch.save(discriminator.state_dict(), save_path / ckpt_name_disc)
            
        wandb.log(log_dict, step=step)

    print(f"Training complete. Saving model at {save_path}")
    torch.save(decoder.state_dict(), save_path / "decoder_trained.pth")
    if discriminator:
        torch.save(discriminator.state_dict(), save_path / "discriminator_trained.pth")
    wandb.finish()
