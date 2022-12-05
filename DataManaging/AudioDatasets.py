import scipy as scipy
import tqdm as tqdm
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import pandas as pd
import librosa
import local_vars
import os
from DatasetCreation import create_audio_dataset


def load_wav_to_torch(path, sampling_rate):
    data, sampling_rate = librosa.core.load(path, sr=sampling_rate)
    data = 0.95 * librosa.util.normalize(data)
    return torch.from_numpy(data).float(), sampling_rate



class AudioDataset(Dataset):

    def __init__(self, annotations, sampling_rate):
        self.sampling_rate = sampling_rate
        # self.segment_length = segment_length

        self.audio_files = annotations['preprocessed_file'].to_numpy()
        self.gender_idx = annotations['gender'].to_numpy()
        self.labels = annotations['label'].to_numpy()
        self.speaker_id = annotations['speaker_id'].to_numpy()

        # self._preprocess()

        # if os.path.exists(data_path):
        #     print("Found pre-processed AudioMNIST...")
        # else:
        #     print("No pre-processed AudioMNIST found. Preprocessing...")
        #     create_audio_dataset(data_path, sampling_rate, segment_length, )
        # data, sr = self._load_preprocessed(data_path)


    def _load_preprocessed_file(self, path):
        data, sampling_rate = librosa.core.load(path, sr=self.sampling_rate)
        data = 0.95 * librosa.util.normalize(data)
        return torch.from_numpy(data).float(), sampling_rate


    def __getitem__(self, item):
        audio_file = self.audio_files[item]
        audio, sampling_rate = load_wav_to_torch(audio_file, self.sampling_rate)
        return audio, self.gender_idx[item], self.labels[item], self.speaker_id[item]

    def __len__(self):
        return len(self.audio_files)


    @staticmethod
    def load(sampling_rate=8000):
        train_annotations = pd.read_csv(local_vars.PREPROCESSED_AUDIO_MNIST_PATH + 'train_annotations.csv')
        trainData = AudioDataset(train_annotations, sampling_rate)

        test_annotations = pd.read_csv(local_vars.PREPROCESSED_AUDIO_MNIST_PATH + 'test_annotations.csv')
        testData = AudioDataset(test_annotations, sampling_rate)

        return trainData, testData
