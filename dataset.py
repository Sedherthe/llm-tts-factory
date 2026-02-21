import json
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, path):
        with open(path, encoding='utf-8') as f:
            self.dataset = json.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text, audio, audio_path = self.dataset[idx]
        # Format: [STOP][TEXT]<text prompt>[START]<audio tokens>[STOP]
        res = f"[TEXT]{text}[START]{''.join(list(map(lambda x: f'[{x}]', audio)))}[STOP]"
        return res
