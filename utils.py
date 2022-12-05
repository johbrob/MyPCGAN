from pathlib import Path
from enum import Enum
import scipy.io


class Mode(Enum):
    TRAIN = 0
    EVAL = 1


def set_mode(models, mode):
    frozen_model_counter = 0
    if mode == Mode.TRAIN:
        for name, model in models.items():
            if name == 'digit_classifier' or name == 'gender_classifier':
                frozen_model_counter += 1
            else:
                model.train()
    elif mode == Mode.EVAL:
        for name, model in models.items():
            if name == 'digit_classifier' or name == 'gender_classifier':
                frozen_model_counter += 1
            else:
                model.eval()

    # make sure we don't accidentally set frozen classifiers to train mode'
    assert frozen_model_counter == 2

    return models


def save_sample(file_path, sampling_rate, audio):
    if isinstance(file_path, str):
        file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    audio = (audio.numpy() * 32768).astype("int16")
    scipy.io.wavfile.write(file_path, sampling_rate, audio)