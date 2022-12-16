from abc import abstractmethod
from torch.utils.data import Dataset
from datasets import data_utils
import pandas as pd
import numpy as np
import librosa
import torch
import enum
import os


class Gender(enum.Enum):
    FEMALE = 0,
    MALE = 1


def _load_wav_to_torch(path, sampling_rate):
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
        audio, sampling_rate = _load_wav_to_torch(audio_file, self.sampling_rate)
        return audio, self.gender_idx[item], self.labels[item], self.speaker_id[item], self.audio_files[item]

    def __len__(self):
        return len(self.audio_files)

    @classmethod
    def load(cls, sampling_rate=None, segment_length=None, even_gender_proportions=None):
        if sampling_rate is not None or segment_length is not None or even_gender_proportions is not None:
            raise NotImplementedError("No support for custom settings for now")

        if not os.path.exists(cls.get_save_path()):
            print('No preprocessed data found...')
            if os.path.exists(cls.get_load_path()):
                print(f'Raw dataset found...Start preprocessing with sample rate {cls.get_default_sampling_rate()} '
                      f'and segment_length {cls.get_default_segment_length()}')
                cls._create_dataset()
            else:
                print('No raw dataset found...Make sure dataset is available')

        prefix = 'even_' if even_gender_proportions is None or even_gender_proportions else ''
        save_path = cls.get_save_path() if cls.get_save_path()[-1] == '/' else cls.get_save_path() + '/'

        train_annotations = pd.read_csv(save_path + prefix + 'train_annotations.csv')
        trainData = cls(train_annotations, cls.get_default_sampling_rate(), cls.get_default_segment_length())

        test_annotations = pd.read_csv(save_path + prefix + 'test_annotations.csv')
        testData = cls(test_annotations, cls.get_default_sampling_rate(), cls.get_default_segment_length())

        return trainData, testData

    @staticmethod
    @abstractmethod
    def _create_dataset(sampling_rate=None, segment_length=None, test_split_ratio=None, even_gender_proportions=None) -> None:
        pass

    @staticmethod
    @abstractmethod
    def get_load_path() -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_save_path() -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        pass

    @staticmethod
    def get_default_sampling_rate() -> int:
        pass

    @staticmethod
    def get_default_segment_length() -> int:
        pass
