import os
import random
import torch
import librosa
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class ESDataset(Dataset):
    
    def __init__(self, config):

        self._EMO = {'angry': 0, 'happy': 1, 'neutral': 2, 'sad': 3, 'surprise': 4}
        self._ID2EMO = {key: value for value, key in self._EMO.items()}
        self.all_samples = []
        self.samples = {
            'angry': [],
            'happy': [],
            'neutral': [],
            'sad': [],
            'surprise': []
        }
        
        for emo in config['dirs'].keys():
            
            emo_dir = config['dirs'][emo]
            if not os.path.isdir(emo_dir):
                raise FileNotFoundError(f'Cannot find the given directory: {emo_dir}')
            
            files = os.listdir(emo_dir)
            for f in files:
                fpath = os.path.join(emo_dir, f)
                if not os.path.isfile(fpath):
                    continue
                self.all_samples.append((fpath, self._EMO[emo]))
                self.samples[emo].append(fpath)

        # Suffle all samples
        if config.shuffle:
            random.shuffle(self.all_samples)
        
        self.loader = DataLoader(
            self, 
            batch_size=config.batch_size, 
            shuffle=config.shuffle,
            num_workers=config.num_workers,
            collate_fn=Collate(config.sr),
        )

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        anchor, emo = self.all_samples[idx]
        emo_label = self._ID2EMO[emo]

        # Choose 4 samples randomly have same emotion with anchor
        pos_samples = list(random.choices(self.samples[emo_label], k=4))

        # Random one sample per each of other's emotions
        neg_samples = []
        for neg_emo in (self.samples.keys() - [self._ID2EMO[emo]]):
            sample_idx = np.random.randint(0, len(self.samples[neg_emo]))
            sample = self.samples[neg_emo][sample_idx]
            neg_samples.append(sample)

        
        return anchor, pos_samples, neg_samples

    
class Collate:
    
    def __init__(self, sr):
        self.sr = sr
        
    def __call__(self, batch):
        
        max_length = 0
        signal_list = []
        
        for anchor, pos_samples, neg_samples in batch:

            anchor, _ = librosa.load(anchor, sr=self.sr)
            pos_samples, _ = zip(*[librosa.load(pos_sample, sr=self.sr) for pos_sample in pos_samples])
            neg_samples, _ = zip(*[librosa.load(neg_sample, sr=self.sr) for neg_sample in neg_samples])
            
            signal_list.append(torch.tensor(anchor))
            signal_list = signal_list + [torch.tensor(sample) for sample in pos_samples]
            signal_list = signal_list + [torch.tensor(sample) for sample in neg_samples]

            max_length = max(max_length, 
                             len(anchor), 
                             len(max(pos_samples, key=len)), 
                             len(max(neg_samples, key=len)))
        

        for idx in range(signal_list):
            sample_len = signal_list[idx].size(0)
            pad = (0, max_length - sample_len)
            signal_list[idx] = F.pad(signal_list[idx], pad)

        signal_list = torch.stack(signal_list)    
        
        return signal_list

