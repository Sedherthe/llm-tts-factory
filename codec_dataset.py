import torch
import torchaudio
from torch.utils.data import Dataset
import os
import json


class LJSpeechDataset(Dataset):
    def __init__(self, root, sample_rate=32000,  mode='train'):
        """
        root: path to LJSpeech-1.1 directory
        """
        self.root = root
        self.sample_rate = sample_rate

        # Write code here to handle modes. Take wave files only from mode json. 
        self.mode = mode
        mode_json = os.path.join(root, f"{mode}.json")
        with open(mode_json, 'r') as f:
            self.dataset = json.load(f)

        # self.wav_dir = os.path.join(root, "wavs")
        # self.wav_files = sorted(
        #     [f for f in os.listdir(self.wav_dir) if f.endswith(".wav")]
        # )

        # assert len(self.wav_files) > 0, "No wav files found!"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # wav_path = os.path.join(self.wav_dir, self.wav_files[idx])
        
        # Using train and val json files
        item = self.dataset[idx]
        text, audio_tokens, wav_path = item

        wav, sr = torchaudio.load(wav_path)

        # mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # resample if needed
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(
                wav, orig_freq=sr, new_freq=self.sample_rate
            )

        # print("wav shape is: ", wav.shape, idx)

        return wav
