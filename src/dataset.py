# src/dataset.py

import os
import torch
import random
import soundfile
import numpy as np
import glob
from torch.utils.data import Dataset
from scipy import signal
import speechbrain as sb


# # 变长
# for reading the train split, we add chunking
def audio_pipeline_for_unfixed_length(file_path, cut_length_interval_in_frames):
    """Load the signal, and pass it and its length to the corruption class.
    This is done on the CPU in the `collate_fn`."""
    sig = sb.dataio.dataio.read_audio(file_path)
    if cut_length_interval_in_frames is not None:
        cut_length = random.randint(*cut_length_interval_in_frames)
        # pick the start index of the cut
        left_index = random.randint(0, len(sig) - cut_length)
        # cut the signal
        sig = sig[left_index : left_index + cut_length]
    return sig

# 定长
def audio_pipeline_for_fixed_length(file_path, max_audio):
    """Load the signal, and pass it and its length to the corruption class.
    This is done on the CPU in the `collate_fn`."""
    sig = sb.dataio.dataio.read_audio(file_path)
    # padding
    if sig.shape[0] <= max_audio:
        shortage = max_audio - sig.shape[0]
        sig = np.pad(sig, (0, shortage), 'wrap')

    strat_frame = np.int64(random.random()*(sig.shape[0]-max_audio))
    sig = sig[strat_frame:strat_frame+max_audio]

    return sig

class AudioDataset(Dataset):
    def __init__(self, dataframe, args, split = 'train'):
        self.dataframe = dataframe[dataframe.split==split] # Filter dataframe based on split
        ids = self.dataframe.id
        unique_ids = list(set(ids)) # Get unique IDs
        self.id_map = {id: i for i, id in enumerate(unique_ids)} # Create a mapping of IDs to indices

        self.args = args
        self.split = split

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        if self.args.max_audio is not None:
            audio_pipeline = audio_pipeline_for_fixed_length
        else:
            audio_pipeline = audio_pipeline_for_unfixed_length
        # audio_pipeline read the audio file and returns a random crop
        audio_item = audio_pipeline(self.dataframe.iloc[index]['file_path'], self.dataframe.iloc[index]['cut_length_interval_in_frames'] )
        # Get the label corresponding to the ID from the id_map
        label = self.id_map[self.dataframe.iloc[index]['id']]    
        return audio_item, label

    def shuffle_dataframe(self):
        # Shuffle the dataframe and update the ID mapping
        self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)
        self.ids = self.dataframe.id
        unique_ids = list(set(self.ids))
        self.id_map = {id: i for i, id in enumerate(unique_ids)}

def collate_fn(batch):
    # Unpack the batch into waveforms and labels
    waveforms, labels = zip(*batch)
    waveforms = [waveform.squeeze() for waveform in waveforms]  # Squeeze waveforms
    waveforms = [torch.tensor(waveform).float() for waveform in waveforms] # Convert waveforms to tensors
    waveforms_padded = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True) # Pad waveforms
    waveforms_lens = [waveform.shape[-1]/waveforms_padded[0].shape[-1] for waveform in waveforms] # Compute waveform lengths
    return waveforms_padded, waveforms_lens, labels

class Sampler():
    # custom sampler that used to generate batches of random indices for data shuffling and batch sampling in the DataLoader.
    def __init__(self, batch_size, data_len):
        self.batch_size = batch_size
        self.data_len = data_len//batch_size
    def __iter__(self):
      # Generate random indices for the batch.
      # Generates random indices for each batch within the range determined by the current batch index i
      # Each batch can contain multiple occurrences of the same baby_id, these cases are considered 'positive' in the triplet loss.
        batch_idxs = []
        for i in range(self.data_len):
          rnd_indeces = list(random.choices(range(i*(self.batch_size), (i+1)*self.batch_size), k=self.batch_size))
          batch_idxs.append(rnd_indeces)
        return iter(batch_idxs)