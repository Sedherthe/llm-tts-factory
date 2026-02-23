"""
Training script for Soprano.

Usage:
python train_llm.py --input-dir path/to/files --save-dir path/to/weights

Args:
--input-dir: Path to directory of LJSpeech-style dataset. If none is provided this defaults to the provided example dataset.
--save-dir: Path to directory to save weights

Adapted from https://github.com/karpathy/nanoGPT
"""
import os
import argparse
import pathlib
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors.torch import load_file
import wandb

from dataset import AudioDataset

# -----------------------------------------------------------------------------
# Global Configuration
# -----------------------------------------------------------------------------
DEVICE = 'cuda:0'
SEED = 1337

MAX_LR = 2e-5
WARMUP_RATIO = 0.3
COOLDOWN_RATIO = 0.1
MIN_LR = 0.3 * MAX_LR
BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 1
SEQ_LEN = 1024
TEXT_FACTOR = 0.5  # currently does not train on text inputs, you can increase to change this
MAX_STEPS = 150000
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
START_STEP = 0

VAL_FREQ = 250
SAVE_FREQ = 5000

# Pre-calculated derived config
WARMUP_STEPS = int(MAX_STEPS * WARMUP_RATIO)
COOLDOWN_STEPS = int(MAX_STEPS * COOLDOWN_RATIO)

# Options
WANDB_PROJECT = "soprano-llm"
TOKENIZER_NAME = 'ekwek/Soprano-80M'
PRETRAINED_CKPT_PATH = "/home/ubuntu/soma/ckpt/suprano/suprano_llm/codec_v2/v2/checkpoint-40000/model.safetensors"
# -----------------------------------------------------------------------------


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
    parser.add_argument("--from-scratch",
        action="store_true",
        help="If set, train from scratch using the config, instead of loading pretrained weights."
    )
    return parser.parse_args()


def worker_seed_init(_):
    worker_seed = torch.initial_seed() % (2**32-1)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_lr(it): # WSD schedule
    if it < WARMUP_STEPS:
        return MAX_LR * (it + 1) / WARMUP_STEPS
    if it < MAX_STEPS - COOLDOWN_STEPS:
        return MAX_LR
    return MIN_LR + (MAX_LR - MIN_LR) * ((MAX_STEPS - it) / COOLDOWN_STEPS)


def collate_pack(texts, tokenizer):
    tokens_batch = tokenizer(texts, padding=False, truncation=False)
    batch = []
    cur_sample, cur_size = [], 0
    for i in range(len(texts)):
        tokens = torch.tensor(tokens_batch['input_ids'][i][:-1], dtype=torch.long)
        cur_size += tokens.size(0)
        cur_sample.append(tokens)
        if cur_size >= SEQ_LEN + 1:
            batch.append(torch.cat(cur_sample)[: SEQ_LEN + 1])
            cur_sample, cur_size = [], 0
            if len(batch) == BATCH_SIZE:
                break
    if cur_sample and not batch: # add partial sample if there isn't enough data
        batch_item = torch.cat(cur_sample + [torch.zeros(SEQ_LEN, dtype=torch.long)])[: SEQ_LEN + 1]
        batch.append(batch_item)
    if len(batch) < BATCH_SIZE:
        # pad up to batch_size for consistency
        pad = batch[-1]
        while len(batch) < BATCH_SIZE:
            batch.append(pad)
    batch = torch.stack(batch)
    x = batch[:, :-1]
    y = batch[:, 1:]
    return x, y


def collate_dynamic(texts, tokenizer):
    # Dynamic Batching: Pad to the longest in this batch (max 2048 safety)
    tokenized = tokenizer(texts, padding=True, truncation=True, max_length=2048, return_tensors='pt', add_special_tokens=False)
    batch = tokenized['input_ids']
    attn_mask = tokenized['attention_mask']
    
    x = batch[:, :-1]
    y = batch[:, 1:]
    # Attention mask needs to align with x. Since we shift x by removing the last token,
    # we should also remove the last token from the mask.
    attn_mask = attn_mask[:, :-1]
    
    return x, y, attn_mask


def collate_pack_val(texts, tokenizer):
    # max_length=SEQ_LEN+1 because we need to shift for x, y
    out = tokenizer(texts, padding=True, truncation=True, max_length=SEQ_LEN+1, return_tensors='pt', add_special_tokens=False)
    batch = out['input_ids']
    
    # Ensure fixed length padding to SEQ_LEN + 1
    if batch.size(1) < SEQ_LEN + 1:
        pad_len = SEQ_LEN + 1 - batch.size(1)
        batch = torch.nn.functional.pad(batch, (0, pad_len), value=tokenizer.pad_token_id)
        
    x = batch[:, :-1]
    y = batch[:, 1:]
    return x, y


def compute_loss(x, logits, y, num_steps, mask=None):
    pred = logits.view(-1, logits.size(-1))
    labels = y.reshape(-1)
    loss = torch.nn.functional.cross_entropy(pred, labels, reduction='none')
    
    if mask is not None:
        mask = mask.reshape(-1)
        loss = loss * mask
    
    # Audio tokens: >=3 and <=8003. 
    # NOTE: If [STOP] is 3, it counts as audio.
    # We apply the mask to filter out padding.
    audio_mask_cond = torch.logical_and(labels >= 3, labels <= 8003)
    if mask is not None:
        audio_mask = audio_mask_cond & (mask > 0)
    else:
        audio_mask = audio_mask_cond
    
    # Text tokens: The rest, BUT excluding masked (padding) tokens
    if mask is not None:
        text_mask = (~audio_mask_cond) & (mask > 0)
    else:
        text_mask = ~audio_mask_cond

    # Avoid division by zero
    audio_mean = loss[audio_mask].mean() if audio_mask.sum() > 0 else torch.tensor(0.0, device=loss.device)
    text_mean = loss[text_mask].mean() if text_mask.sum() > 0 else torch.tensor(0.0, device=loss.device)
    
    # Acc: only on non-masked tokens. 
    # Current logic: (logits.argmax(dim=-1) == y).view(-1)[audio_mask]
    # This correctly calculates accuracy only on valid audio tokens.
    acc = (logits.argmax(dim=-1).view(-1) == labels).view(-1)[audio_mask].to(torch.float32).mean()
    if torch.isnan(acc): acc = torch.tensor(0.0, device=loss.device)

    audio_loss = audio_mean / num_steps
    text_loss = text_mean / num_steps
    acc = acc / num_steps
    return audio_loss, text_loss, acc


def evaluate(model, val_dataloader, step, device):
    model.eval()
    val_dataloader_it = iter(val_dataloader)
    with torch.no_grad():
        val_audio_loss_accum = torch.tensor(0.0).to(device)
        val_text_loss_accum = torch.tensor(0.0).to(device)
        val_acc_accum = torch.tensor(0.0).to(device)
        val_loss_steps = len(val_dataloader)
        for _ in range(val_loss_steps):
            x, y, attn_mask = next(val_dataloader_it)
            x, y, attn_mask = x.to(device), y.to(device), attn_mask.to(device)
            logits = model(x, attention_mask=attn_mask).logits
            audio_loss, text_loss, acc = compute_loss(x, logits, y, val_loss_steps, mask=attn_mask)
            val_audio_loss_accum += audio_loss.detach()
            val_text_loss_accum += text_loss.detach()
            val_acc_accum += acc.detach()
            
        print(f"validation text loss: {val_text_loss_accum.item():.4f}\tvalidation audio loss: {val_audio_loss_accum.item():.4f}\tvalidation acc: {val_acc_accum.item():.4f}")
        wandb.log({
            "val/text_loss": val_text_loss_accum.item(),
            "val/audio_loss": val_audio_loss_accum.item(),
            "val/acc": val_acc_accum.item()
        }, step=step)
    model.train()


def main():
    args = get_args()

    train_dataset_path = f'{args.input_dir}/train.json'
    val_dataset_path = f'{args.input_dir}/val.json'
    save_path = args.save_dir
    os.makedirs(save_path, exist_ok=True)

    device_type = "cuda" if DEVICE.startswith("cuda") else "cpu"
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    torch.set_float32_matmul_precision('high')
    
    print(f"Save Path: {save_path}")

    wandb.init(project=WANDB_PROJECT, config=vars(args))

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.padding_side = 'right' # Essential for training!

    # model
    if args.from_scratch:
        print("Initializing model from scratch (random weights)...")
        config = AutoConfig.from_pretrained(TOKENIZER_NAME)
        model = AutoModelForCausalLM.from_config(config)
    else:
        print("Loading pretrained model weights...")
        model = AutoModelForCausalLM.from_pretrained(TOKENIZER_NAME)
    
    if not args.from_scratch:
        state_dict = load_file(PRETRAINED_CKPT_PATH)
        model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.train()

    # dataset
    dataset = AudioDataset(train_dataset_path)
    
    # Using dynamic batching now
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_seed_init,
        collate_fn=lambda texts: collate_dynamic(texts, tokenizer),
    )
    dataloader_it = iter(dataloader)
    
    val_dataset = AudioDataset(val_dataset_path)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_seed_init,
        collate_fn=lambda texts: collate_dynamic(texts, tokenizer),
    )

    import pdb;pdb.set_trace()

    # optimizer
    opt = torch.optim.AdamW(model.parameters(), MAX_LR, betas=BETAS, weight_decay=WEIGHT_DECAY, fused=True)

    pbar = tqdm(range(START_STEP, MAX_STEPS), ncols=200, dynamic_ncols=True)
    for step in pbar:
        start = time.time()
        
        if VAL_FREQ > 0 and step != 0 and (step % VAL_FREQ == 0 or step == MAX_STEPS - 1):
            evaluate(model, val_dataloader, step, DEVICE)
        
        if SAVE_FREQ > 0 and step % SAVE_FREQ == 0:
            ckpt_path = os.path.join(save_path, f"checkpoint-{step}")
            print(f"Saving checkpoint to {ckpt_path}")
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)

        opt.zero_grad()
        audio_loss_accum = 0.0
        text_loss_accum = 0.0
        acc_accum = 0.0
        
        for micro_step in range(GRAD_ACCUM_STEPS):
            try:
                x, y, attn_mask = next(dataloader_it)
            except StopIteration:
                dataloader_it = iter(dataloader)
                x, y, attn_mask = next(dataloader_it)
                
            x, y, attn_mask = x.to(DEVICE), y.to(DEVICE), attn_mask.to(DEVICE)

            logits = model(x, attention_mask=attn_mask).logits
            audio_loss, text_loss, acc = compute_loss(x, logits, y, GRAD_ACCUM_STEPS, mask=attn_mask)
            
            audio_loss_accum += audio_loss.detach()
            text_loss_accum += text_loss.detach()
            acc_accum += acc.detach()
            
            total_loss = audio_loss + TEXT_FACTOR * text_loss
            total_loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in opt.param_groups:
            param_group['lr'] = lr
        opt.step()
        torch.cuda.synchronize()
        
        total_tokens = step * BATCH_SIZE * SEQ_LEN * GRAD_ACCUM_STEPS
        end = time.time()
        dt = (end - start) * 1000
        tokens_per_second = (BATCH_SIZE * SEQ_LEN * GRAD_ACCUM_STEPS) / (end - start)
        
        tqdm_log = f'text loss: {text_loss_accum.item():.3f} | audio loss: {audio_loss_accum.item():.3f} | acc: {acc_accum.item():.4f} | lr: {lr:.2e} | norm: {norm:.3f} | time: {dt:.2f} ms | {tokens_per_second:.2f} t/s'
        pbar.set_description(tqdm_log)
        
        wandb.log({
            "train/text_loss": text_loss_accum.item(),
            "train/audio_loss": audio_loss_accum.item(),
            "train/acc": acc_accum.item(),
            "train/lr": lr,
            "train/grad_norm": norm,
            "train/dt": dt,
            "train/tokens_per_sec": tokens_per_second
        }, step=step)

    print(f"Training complete. Saving model at {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Saving done.")
    wandb.finish()


if __name__ == '__main__':
    main()