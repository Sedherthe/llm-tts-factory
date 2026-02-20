"""
Converts a dataset in LJSpeech format into audio tokens that can be used to train/fine-tune Soprano.
This script creates two JSON files for train and test splits in the provided directory.

Usage:
python generate_dataset.py --input-dir path/to/files

Args:
--input-dir: Path to directory of LJSpeech-style dataset. If none is provided this defaults to the provided example dataset.
"""
import argparse
import pathlib
import random
import json


import torchaudio
import torch
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from encoder.codec import Encoder


SAMPLE_RATE = 32000
SEED = 42
VAL_PROP = 0.1
VAL_MAX = 512

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",
        required=False,
        default="./example_dataset",
        type=pathlib.Path
    )
    return parser.parse_args()

def main():
    args = get_args()
    input_dir = args.input_dir

    use_custom_model = True

    print("Loading model.")
    encoder = Encoder()

    if not use_custom_model:
        encoder_path = hf_hub_download(repo_id='ekwek/Soprano-Encoder', filename='encoder.pth')
        encoder.load_state_dict(torch.load(encoder_path))
    else:


        speech_autoencoder_path = "/home/ubuntu/soma/ckpt/suprano/suprano_codec/codec_1/step_40000.pt"
        print("Loading model using custom model path!", speech_autoencoder_path)

        full_ckpt = torch.load(speech_autoencoder_path)
        encoder_state_dict = {k.replace("encoder.", ""): v for k, v in full_ckpt.items() if k.startswith("encoder.")}

        encoder_state_dict = {}
        for k,v in full_ckpt.items():
            if k.startswith("encoder."):
                # replace the first occurance of 'encoder.' only 
                new_k = k.replace("encoder.", "", 1)
                encoder_state_dict[new_k] = v

        encoder.load_state_dict(encoder_state_dict)
    print("Model loaded.")

    import pdb;pdb.set_trace()


    print("Reading metadata.")
    files = []
    with open(f'{input_dir}/metadata_orig.csv', encoding='utf-8') as f:
        data = f.read().split('\n')
        for line in data:

            # import pdb;pdb.set_trace()

            out = line.split("|", maxsplit=1)
            filename = out[0]
            transcript = out[-1].split('|')[-1]
            files.append((filename, transcript))

    print(f'{len(files)} samples located in directory.')

    import pdb;pdb.set_trace()

    print("Encoding audio.")
    dataset = []
    for sample in tqdm(files):
        filename, transcript = sample
        # sr, audio = wavfile.read(f'{input_dir}/wavs/{filename}.wav')
        # audio = torch.from_numpy(audio)

        import pdb;pdb.set_trace()

        try:
            audio, sr = torchaudio.load(f'{input_dir}/wavs/{filename}.wav')
        except:
            print("Error loading audio: ", filename)
            continue
        # import pdb;pdb.set_trace()

        if sr != SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
        # audio = audio.unsqueeze(0)
        with torch.no_grad():
            audio_tokens = encoder(audio)
        # Save absolute path to audio for loading in training
        audio_path = str(pathlib.Path(f'{input_dir}/wavs/{filename}.wav').resolve())
        dataset.append([transcript, audio_tokens.squeeze(0).tolist(), audio_path])

    print("Generating train/test splits.")
    random.seed(SEED)
    random.shuffle(dataset)
    num_val = min(int(VAL_PROP * len(dataset)) + 1, VAL_MAX)
    train_dataset = dataset[num_val:]
    val_dataset = dataset[:num_val]
    print(f'# train samples: {len(train_dataset)}')
    print(f'# val samples: {len(val_dataset)}')

    print("Saving datasets.")
    with open(f'{input_dir}/train.json', 'w', encoding='utf-8') as f:
        json.dump(train_dataset, f, indent=2)
    with open(f'{input_dir}/val.json', 'w', encoding='utf-8') as f:
        json.dump(val_dataset, f, indent=2)
    print("Datasets saved.")


if __name__ == '__main__':
    main()
