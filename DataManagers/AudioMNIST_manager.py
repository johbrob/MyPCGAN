import numpy as np
import pandas as pd
import librosa


def build_annotation_index(training_files, annotation_file, balanced_genders=False):
    annotations = pd.read_json(annotation_file, orient='index')
    training_files = librosa.util.find_files(training_files)

    ids = [int(f.split('/')[-2]) for f in training_files]
    digits = np.array([int(f.split('/')[-1][0]) for f in training_files])

    gender_idx = np.array([int(annotations.loc[i]['gender'] == 'male') for i in ids])
    num_speakers = len(annotations.index)

    return training_files, gender_idx, digits, ids
