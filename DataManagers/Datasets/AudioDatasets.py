import scipy as scipy
import tqdm as tqdm
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import librosa.core.load

def save_sample(file, sampling_rate, audio):
    audio = (audio.numpy() * 32768).astype("int16")
    scipy.io.wavfile.write(file, sampling_rate, audio)

class AudioDataset(Dataset):

    def __init__(self, an_idx, sampling_rate, segment_length):
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length

        self.audio_files = an_idx['filename'].to_numpy()
        self.gender_idx = an_idx['gender'].to_numpy()
        self.speaker_id = an_idx['speaker_id'].to_numpy()

        self._preprocess()


    def _preprocess(self):

        for file in tqdm.tqdm(self.audio_files):
            audio, sampling_rate = librosa.core.load(file, sr=self.sampling_rate)
            audio = torch.from_numpy(audio).float()

            if audio.shape[0] > 8192:
                print('Corrupted audio file')
                print(f"has length {audio.shape[0]}")

            if audio.shape[0] >= self.segment_length:
                audio = audio[:self.segment_length]
            else:
                n_pads = self.segment_length - audio.shape[0]
                if n_pads % 2 == 0:
                    pad1d = (n_pads // 2, n_pads // 2)
                else:
                    pad1d = (n_pads // 2, n_pads // 2 + 1)
                audio = F.pad(audio, pad1d, "constant")

            save_sample(file, sampling_rate, audio)

    def __getitem__(self, item):
        pass