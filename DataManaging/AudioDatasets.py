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


def save_sample(file, sampling_rate, audio):
    audio = (audio.numpy() * 32768).astype("int16")
    scipy.io.wavfile.write(file, sampling_rate, audio)


class AudioDataset(Dataset):

    def __init__(self, annotations):
        # self.sampling_rate = sampling_rate
        # self.segment_length = segment_length

        self.audio_files = annotations['file'].to_numpy()
        self.gender_idx = annotations['gender'].to_numpy()
        self.speaker_id = annotations['speaker_id'].to_numpy()

        # self._preprocess()

        # if os.path.exists(data_path):
        #     print("Found pre-processed AudioMNIST...")
        # else:
        #     print("No pre-processed AudioMNIST found. Preprocessing...")
        #     create_audio_dataset(data_path, sampling_rate, segment_length, )
        # data, sr = self._load_preprocessed(data_path)


    def _load_preprocessed(self, path):
        data, sampling_rate = librosa.core.load(path, sr=self.sampling_rate)
        data = 0.95 * librosa.util.normalize(data)
        return torch.from_numpy(data).float(), sampling_rate


    def __getitem__(self, item):

        pass


    @staticmethod
    def load():
        train_annotations = pd.read_csv(local_vars.PREPROCESSED_AUDIO_MNIST_PATH + 'train_annoations.csv')
        trainData = AudioDataset(train_annotations)

        test_annotations = pd.read_csv(local_vars.PREPROCESSED_AUDIO_MNIST_PATH + 'test_annoations.csv')
        testData = AudioDataset(test_annotations)

        return trainData, testData
