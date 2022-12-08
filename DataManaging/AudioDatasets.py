import utils
from dataset_creation import create_audio_dataset
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import local_vars
import librosa
import torch
import os


def load_wav_to_torch(path, sampling_rate):
    data, sampling_rate = librosa.core.load(path, sr=sampling_rate)
    data = 0.95 * librosa.util.normalize(data)
    return torch.from_numpy(data).float(), sampling_rate


class AudioDataset(Dataset):

    def __init__(self, annotations, sampling_rate, segment_length):
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length

        self.audio_files = annotations['preprocessed_file'].to_numpy()
        self.gender_idx = annotations['gender'].to_numpy()
        self.labels = annotations['label'].to_numpy()
        self.speaker_id = annotations['speaker_id'].to_numpy()

        self.n_genders = len(np.unique(self.gender_idx))
        self.n_labels = len(np.unique(self.labels))
        self.n_speakers = len(np.unique(self.speaker_id))

    def _load_preprocessed_file(self, path):
        data, sampling_rate = librosa.core.load(path, sr=self.sampling_rate)
        data = 0.95 * librosa.util.normalize(data)
        return torch.from_numpy(data).float(), sampling_rate

    def __getitem__(self, item):
        audio_file = self.audio_files[item]
        audio, sampling_rate = load_wav_to_torch(audio_file, self.sampling_rate)
        return audio, self.gender_idx[item], self.labels[item], self.speaker_id[item], self.audio_files[item]

    def __len__(self):
        return len(self.audio_files)

    @staticmethod
    def load(sampling_rate=8000, segment_length=8192):
        if not os.path.exists(local_vars.PREPROCESSED_AUDIO_MNIST_PATH):
            print('No preprocessed data found...')
            if os.path.exists(local_vars.AUDIO_MNIST_PATH):
                print('Raw dataset found...Start preprocessing')
                create_audio_dataset(local_vars.AUDIO_MNIST_PATH,
                                     sampling_rate, segment_length,
                                     local_vars.PREPROCESSED_AUDIO_MNIST_PATH,
                                     0.10)
            else:
                print('No raw dataset found...Make sure dataset is available')

        train_annotations = pd.read_csv(local_vars.PREPROCESSED_AUDIO_MNIST_PATH + 'train_annotations.csv')
        trainData = AudioDataset(train_annotations, sampling_rate, segment_length)

        test_annotations = pd.read_csv(local_vars.PREPROCESSED_AUDIO_MNIST_PATH + 'test_annotations.csv')
        testData = AudioDataset(test_annotations, sampling_rate, segment_length)

        return trainData, testData
