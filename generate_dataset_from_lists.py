"""
Converts a dataset in LJSpeech format into audio tokens for Soprano, using pre-defined train/val lists.

Usage:
python generate_dataset_from_lists.py --input-dir /home/ubuntu/soma/data/lj_speech/LJSpeech-1.1 --output-dir ./dataset_lists
"""
import argparse
import pathlib
import json
import os
import torchaudio
import torch
from tqdm import tqdm
from encoder.codec import Encoder

SAMPLE_RATE = 32000

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",
        required=True,
        type=pathlib.Path,
        help="Path to LJSpeech-1.1 directory containing wavs/, metadata.csv, train_list.txt, val_list.txt"
    )
    parser.add_argument("--output-dir",
        required=True,
        type=pathlib.Path,
        help="Directory to save new train.json and val.json"
    )
    return parser.parse_args()

def load_metadata(input_dir):
    print("Reading metadata.")
    meta_map = {}
    # Check for metadata.csv or metadata_orig.csv
    meta_path = input_dir / 'metadata_orig.csv'
    if not meta_path.exists():
        meta_path = input_dir / 'metadata.csv'
    
    with open(meta_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            filename = parts[0]
            transcript = parts[-1] # formatting might vary, usually last field
            meta_map[filename] = transcript
    return meta_map

def process_list(list_file, meta_map, encoder, input_dir):
    dataset = []
    print(f"Processing {list_file}...")
    with open(list_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    
    for line in tqdm(lines):
        # line is full path: /home/ubuntu/.../wavs/LJxxx.wav
        # Extract filename without extension
        path_obj = pathlib.Path(line)
        filename = path_obj.stem # LJxxx
        
        if filename not in meta_map:
            print(f"Warning: {filename} not found in metadata. Skipping.")
            continue
            
        transcript = meta_map[filename]
        wav_path = str(path_obj)
        
        # Load and Encode
        try:
            audio, sr = torchaudio.load(wav_path)
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            continue

        if sr != SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
            
        # Mono check
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
            
        with torch.no_grad():
            audio_tokens = encoder(audio) # Encoder expects (B, C, T) or (1, T)? 

        # If it returns indices directly. 
        dataset.append([transcript, audio_tokens.squeeze(0).tolist(), wav_path])
        
    return dataset

def main():
    args = get_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load Encoder
    print("Loading Encoder...")
    encoder = Encoder()
    speech_autoencoder_path = "/home/ubuntu/soma/ckpt/suprano/suprano_codec/codec_v2/step_30000.pt"
    
    print(f"Loading weights from {speech_autoencoder_path}")
    full_ckpt = torch.load(speech_autoencoder_path, map_location='cpu')
    
    # Extract encoder weights
    encoder_state_dict = {}
    for k, v in full_ckpt.items():
        if k.startswith("encoder."):
            new_k = k.replace("encoder.", "", 1)
            encoder_state_dict[new_k] = v
            
    encoder.load_state_dict(encoder_state_dict)
    encoder.eval()
    print("Encoder Loaded.")

    # Load Metadata
    meta_map = load_metadata(input_dir)

    # Process Train List
    train_list_path = input_dir / 'train_list.txt'
    if train_list_path.exists():
        train_data = process_list(train_list_path, meta_map, encoder, input_dir)
        with open(output_dir / 'train.json', 'w') as f:
            json.dump(train_data, f, indent=2)
        print(f"Saved {len(train_data)} train samples to {output_dir}/train.json")
    else:
        print(f"Error: {train_list_path} not found.")

    # Process Val List
    val_list_path = input_dir / 'val_list.txt'
    if val_list_path.exists():
        val_data = process_list(val_list_path, meta_map, encoder, input_dir)
        with open(output_dir / 'val.json', 'w') as f:
            json.dump(val_data, f, indent=2)
        print(f"Saved {len(val_data)} val samples to {output_dir}/val.json")
    else:
        print(f"Error: {val_list_path} not found.")

if __name__ == '__main__':
    main()
