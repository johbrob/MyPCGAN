from datasets.audio_dataset import AudioDataset
from datasets import data_utils
import pandas as pd
import numpy as np
import local_vars
import librosa
import os


def create_crema_d(data_path, save_path, test_split_ratio=0.20, segment_length=21048, sampling_rate=8000,
                   even_gender_proportions=False):
    save_path = os.path.join(save_path, CremaD.get_name())
    data_utils.create_dataset(data_path, sampling_rate, segment_length, save_path, _load_raw, test_split_ratio,
                              even_gender_proportions)


sentence_map = {'DFA': 0, 'IEO': 1, 'IOM': 2, 'ITH': 3, 'ITS': 4, 'IWL': 5,
                'IWW': 6, 'MTI': 7, 'TAI': 8, 'TIE': 9, 'TSI': 10, 'WSI': 11}
emotion_map = {'SAD': 0, 'ANG': 1, 'DIS': 2, 'FEA': 3, 'HAP': 4, 'NEU': 5}
emotion_level_map = {'LO': 0, 'MD': 1, 'HI': 2, 'XX': 3}


def _load_raw(data_path):
    data_path = data_path if data_path[-1] == '/' else data_path + '/'
    annotation_file = data_path + 'VideoDemographics.csv'

    annotations = pd.read_csv(annotation_file, index_col='ActorID')
    # gender_idx 'male': 1, 'female': 0
    files = librosa.util.find_files(data_path + 'AudioWAV/')
    file_names = [f.split('/')[-1].split('.')[0] for f in files]
    speaker_ids = [int(f.split('_')[0]) for f in file_names]
    sentence_ids = [sentence_map[f.split('_', 2)[1]] for f in file_names]
    emotion_ids = [emotion_map[f.split('_', 3)[2]] for f in file_names]
    emotion_level_ids = [emotion_level_map[f.split('_', 3)[3]] for f in file_names]

    gender_idx = np.array([int(annotations.loc[i]['Sex'] == 'Male') for i in speaker_ids])

    return pd.DataFrame({'file': files, 'label': sentence_ids, 'gender': gender_idx, 'emotion': emotion_ids,
                         'emotion_level': emotion_level_ids, 'speaker_id': speaker_ids})


class CremaD(AudioDataset):

    def __init__(self, annotations, sampling_rate, segment_length):
        super().__init__(annotations, sampling_rate, segment_length)

    @staticmethod
    def _create_dataset(sampling_rate, segment_length, test_split_ratio, even_gender_proportions):
        return create_crema_d(sampling_rate, segment_length, test_split_ratio, even_gender_proportions)

    @staticmethod
    def get_load_path():
        return local_vars.CREMA_D_PATH

    @staticmethod
    def get_save_path():
        return os.path.join(local_vars.PREPROCESSED_DATA_PATH, 'Crema-D')

    @staticmethod
    def get_name() -> str:
        return 'Crema-D'


if __name__ == '__main__':
    a = CremaD.load()
    print(a)
