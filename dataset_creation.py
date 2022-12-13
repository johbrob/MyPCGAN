import pandas
import torch.nn.functional as F
import local_vars
# import scipy as scipy
import scipy.io
import pandas as pd
import numpy as np
import librosa
import torch
import tqdm
import os


def _load_raw(data_path):
    data_path = data_path if data_path[-1] == '/' else data_path + '/'
    annotation_file = data_path + 'data/audioMNIST_meta.txt'

    annotations = pd.read_json(annotation_file, orient='index')

    # gender_idx 'male': 1, 'female': 0
    training_files = librosa.util.find_files(data_path)
    speaker_ids = [int(f.split('/')[-2]) for f in training_files]
    labels = np.array([int(f.split('/')[-1][0]) for f in training_files])
    gender_idx = np.array([int(annotations.loc[i]['gender'] == 'male') for i in speaker_ids])

    return pd.DataFrame({'file': training_files, 'gender': gender_idx, 'label': labels, 'speaker_id': speaker_ids})


def _balanced_speaker_split(data, test_split_ratio):
    data = data.sort_values(by=['gender', 'speaker_id'])

    # Extract speaker ids for respective gender
    female_speaker_ids = data.loc[data['gender'] == 0].speaker_id.unique()
    male_speaker_ids = data.loc[data['gender'] == 1].speaker_id.unique()

    # Sample speaker IDs according to split ratio
    female_test_ids = np.random.choice(female_speaker_ids, int(len(female_speaker_ids) * test_split_ratio), replace=False)
    male_test_ids = np.random.choice(male_speaker_ids, int(len(male_speaker_ids) * test_split_ratio), replace=False)

    female_train_ids = np.setdiff1d(female_speaker_ids, female_test_ids)
    male_train_ids = np.setdiff1d(male_speaker_ids, male_test_ids)

    test_ids = np.concatenate((female_test_ids, male_test_ids), axis=0)
    train_ids = np.concatenate((female_train_ids, male_train_ids), axis=0)

    test_data = data.loc[data['speaker_id'].isin(test_ids)]
    train_data = data.loc[data['speaker_id'].isin(train_ids)]

    return train_data, test_data


def _save_trimmed_and_padded_audio_files(annotations, sampling_rate, segment_length):
    audio_files = annotations['file'].to_numpy()
    save_paths = annotations['preprocessed_file'].to_numpy()

    for file, save_path in tqdm.tqdm(zip(audio_files, save_paths)):
        audio, sampling_rate = librosa.core.load(file, sr=sampling_rate)
        audio = torch.from_numpy(audio).float()

        if audio.shape[0] > 8192:
            print('Corrupted audio file')
            print(f"has length {audio.shape[0]}")

        if audio.shape[0] >= segment_length:
            audio = audio[:segment_length]
        else:
            n_pads = segment_length - audio.shape[0]
            if n_pads % 2 == 0:
                pad1d = (n_pads // 2, n_pads // 2)
            else:
                pad1d = (n_pads // 2, n_pads // 2 + 1)
            audio = F.pad(audio, pad1d, "constant")

        audio = (audio.numpy() * 32768).astype("int16")
        os.makedirs(save_path.rsplit('/', 1)[0], exist_ok=True)
        scipy.io.wavfile.write(save_path, sampling_rate, audio)


def create_audio_dataset(data_path, sampling_rate, segment_length, save_path, test_split_ratio=0.10):
    annotations = _load_raw(data_path)
    train_annotations, test_annotations = _balanced_speaker_split(annotations, test_split_ratio)
    train_annotations['preprocessed_file'] = [save_path + 'train' + file[len(data_path):] for file in train_annotations['file']]
    test_annotations['preprocessed_file'] = [save_path + 'test' + file[len(data_path):] for file in test_annotations['file']]
    _save_trimmed_and_padded_audio_files(train_annotations, sampling_rate, segment_length)
    _save_trimmed_and_padded_audio_files(test_annotations, sampling_rate, segment_length)

    train_annotations.rename(columns={'file': 'original_file'}, inplace=True)
    test_annotations.rename(columns={'file': 'original_file'}, inplace=True)

    train_annotations.to_csv(save_path + 'train_annotations.csv')
    test_annotations.to_csv(save_path + 'test_annotations.csv')

    pd.DataFrame([[1, 0]], columns=['male', 'female']).to_csv(save_path + 'gender_encoding.csv')

if __name__ == '__main__':
    sampling_rate = 8000
    segment_length = 8192
    load_path = local_vars.AUDIO_MNIST_PATH
    save_path = local_vars.PREPROCESSED_AUDIO_MNIST_PATH
    create_audio_dataset(load_path, segment_length, sampling_rate, save_path)