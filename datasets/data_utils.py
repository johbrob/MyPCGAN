import torch.nn.functional as F
import pandas as pd
import numpy as np
import scipy.io
import librosa
import torch
import tqdm
import os


def balanced_speaker_split(data, test_split_ratio, even_gender_proportions=True):
    data = data.sort_values(by=['gender', 'speaker_id'])

    # Extract speaker ids for respective gender
    female_speaker_ids = data.loc[data['gender'] == 0].speaker_id.unique()
    male_speaker_ids = data.loc[data['gender'] == 1].speaker_id.unique()

    # Sample speaker IDs according to split ratio
    female_test_ids = np.random.choice(female_speaker_ids, int(len(female_speaker_ids) * test_split_ratio),
                                       replace=False)
    male_test_ids = np.random.choice(male_speaker_ids, int(len(male_speaker_ids) * test_split_ratio), replace=False)

    female_train_ids = np.setdiff1d(female_speaker_ids, female_test_ids)
    male_train_ids = np.setdiff1d(male_speaker_ids, male_test_ids)

    if even_gender_proportions:
        male_test_ids = male_test_ids[:len(female_test_ids)]
        male_train_ids = male_train_ids[:len(female_train_ids)]
    print(f'Num females in train/test: {len(female_train_ids)}/{len(female_test_ids)}\n'
          f'Num males in train/test: {len(male_train_ids)}/{len(male_test_ids)}')

    test_ids = np.concatenate((female_test_ids, male_test_ids), axis=0)
    train_ids = np.concatenate((female_train_ids, male_train_ids), axis=0)

    test_data = data.loc[data['speaker_id'].isin(test_ids)]
    train_data = data.loc[data['speaker_id'].isin(train_ids)]

    return train_data, test_data


def save_trimmed_and_padded_audio_files(annotations, sampling_rate, segment_length):
    audio_files = annotations['file'].to_numpy()
    save_paths = annotations['preprocessed_file'].to_numpy()

    for file, save_path in tqdm.tqdm(zip(audio_files, save_paths), total=len(annotations)):
        audio, sampling_rate = librosa.core.load(file, sr=sampling_rate)
        audio = torch.from_numpy(audio).float()

        # if audio.shape[0] > 8192:
        #     print('Corrupted audio file')
        #     print(f"has length {audio.shape[0]}")

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


def create_dataset(data_path, sampling_rate, segment_length, save_path, load_raw_func, test_split_ratio=0.20,
                   even_gender_proportions=False):
    annotations = load_raw_func(data_path)
    train_annotations, test_annotations = balanced_speaker_split(annotations, test_split_ratio, even_gender_proportions)
    train_annotations['preprocessed_file'] = [save_path + 'train' + file[len(data_path):] for file in
                                              train_annotations['file']]
    test_annotations['preprocessed_file'] = [save_path + 'test' + file[len(data_path):] for file in
                                             test_annotations['file']]
    save_trimmed_and_padded_audio_files(train_annotations, sampling_rate, segment_length)
    save_trimmed_and_padded_audio_files(test_annotations, sampling_rate, segment_length)

    train_annotations.rename(columns={'file': 'original_file'}, inplace=True)
    test_annotations.rename(columns={'file': 'original_file'}, inplace=True)

    prefix = 'even_' if even_gender_proportions else ''

    train_annotations.to_csv(save_path + prefix + 'train_annotations.csv')
    test_annotations.to_csv(save_path + prefix + 'test_annotations.csv')

    pd.DataFrame([[1, 0]], columns=['male', 'female']).to_csv(save_path + prefix + 'gender_encoding.csv')
