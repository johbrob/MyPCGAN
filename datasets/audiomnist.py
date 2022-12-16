from datasets.audio_dataset import AudioDataset
from datasets import data_utils
import pandas as pd
import numpy as np
import local_vars
import librosa
import os


def _load_raw_data(data_path):
    data_path = data_path if data_path[-1] == '/' else data_path + '/'
    annotation_file = data_path + 'data/audioMNIST_meta.txt'

    annotations = pd.read_json(annotation_file, orient='index')

    # gender_idx 'male': 1, 'female': 0
    training_files = librosa.util.find_files(data_path)
    speaker_ids = [int(f.split('/')[-2]) for f in training_files]
    labels = np.array([int(f.split('/')[-1][0]) for f in training_files])
    gender_idx = np.array([int(annotations.loc[i]['gender'] == 'male') for i in speaker_ids])

    return pd.DataFrame({'file': training_files, 'gender': gender_idx, 'label': labels, 'speaker_id': speaker_ids})


def create_audiomnist(data_path, save_path, test_split_ratio=0.20, segment_length=8192, sampling_rate=8000,
                      even_gender_proportions=False):
    save_path = os.path.join(save_path, AudioMNIST.get_name())
    data_utils.create_dataset(data_path, sampling_rate, segment_length, save_path, _load_raw_data, test_split_ratio,
                              even_gender_proportions)


class AudioMNIST(AudioDataset):

    def __init__(self, annotations, sampling_rate, segment_length):
        super().__init__(annotations, sampling_rate, segment_length)

    @staticmethod
    def _create_dataset(sampling_rate, segment_length, test_split_ratio, even_gender_proportions):
        return create_audiomnist(AudioMNIST.get_load_path(), AudioMNIST.get_save_path(), test_split_ratio,
                                 segment_length, sampling_rate, even_gender_proportions)

    @staticmethod
    def get_load_path():
        return local_vars.AUDIO_MNIST_PATH

    @staticmethod
    def get_save_path():
        return os.path.join(local_vars.PREPROCESSED_DATA_PATH, 'AudioMNIST')

    @staticmethod
    def get_name() -> str:
        return 'AudioMNIST'
