from pathlib import Path
from enum import Enum
import scipy.io
import plotting
import torch
import glob
import os


class Mode(Enum):
    TRAIN = 0
    EVAL = 1


def set_mode(models, mode):
    frozen_model_counter = 0
    if mode == Mode.TRAIN:
        for name, model in models.items():
            if name == 'label_classifier' or name == 'secret_classifier':
                frozen_model_counter += 1
            else:
                model.train()
    elif mode == Mode.EVAL:
        for name, model in models.items():
            if name == 'label_classifier' or name == 'secret_classifier':
                frozen_model_counter += 1
            else:
                model.eval()

    # make sure we don't accidentally set frozen classifiers to train mode'
    assert frozen_model_counter == 2

    return models


def save_audio_file(file_path, sampling_rate, audio):
    if isinstance(file_path, str):
        file_path = Path(file_path)
    elif not isinstance(file_path, Path):
        raise ValueError('File path must be either str or Path')

    file_path.parent.mkdir(parents=True, exist_ok=True)
    audio = (audio.cpu().numpy() * 32768).astype("int16")
    scipy.io.wavfile.write(file_path, sampling_rate, audio)


def zero_grad(optimizers):
    [optimizer.zero_grad() for optimizer in optimizers.values()]


def step(optimizers):
    [optimizer.step() for optimizer in optimizers.values()]


def backward(losses):
    [loss['final'].backward() for loss in losses.values()]


def nestedConfigs2dict(nested_config):
    return {k: v for attr in nested_config.__dict__.values() for k, v in vars(attr).items()}


def save_sample(example_dirs, id, label, epoch, pred_label_male, pred_label_female, filtered_audio, audio_male,
                audio_female, original_audio, sampling_rate):

    speaker_digit_str = f'speaker_{id.item()}_digit_{label.item()}'
    speaker_digit_epoch_str = speaker_digit_str + f'_epoch_{epoch}'
    build_str = lambda gender, digit: speaker_digit_epoch_str + f'_sampled_gender_{gender}_predicted_digit_{digit}.wav'

    filtered_audio_file = os.path.join(example_dirs['audio'], speaker_digit_epoch_str + '_filtered.wav')
    male_audio_file = os.path.join(example_dirs['audio'], build_str('male', pred_label_male.item()))
    female_audio_file = os.path.join(example_dirs['audio'], build_str('female', pred_label_female.item()))
    original_audio_file = os.path.join(example_dirs['audio'], speaker_digit_str + '.wav')

    save_audio_file(filtered_audio_file, sampling_rate, filtered_audio.cpu())
    save_audio_file(male_audio_file, sampling_rate, audio_male.cpu())
    save_audio_file(female_audio_file, sampling_rate, audio_female.cpu())
    save_audio_file(original_audio_file, sampling_rate, original_audio.cpu())


def save_models_and_optimizers(checkpoint_dir, epoch, models, optimizers):
    old_checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, '*latest*')))
    if old_checkpoints:
        for i, _ in enumerate(old_checkpoints):
            os.remove(old_checkpoints[i])
    for k, v in models.items():
        if k == 'label_classifier' or k == 'secret_classifier':
            continue
        torch.save(v.state_dict(), os.path.join(checkpoint_dir, f'{k}_epoch_{epoch}.pt'))
        torch.save(v.state_dict(), os.path.join(checkpoint_dir, f'{k}_latest_epoch_{epoch}.pt'))
    for k, v in optimizers.items():
        torch.save(v.state_dict(), os.path.join(checkpoint_dir, f'optimizer_{k}_epoch_{epoch}.pt'))
        torch.save(v.state_dict(), os.path.join(checkpoint_dir, f'optimizer_{k}_latest_epoch_{epoch}.pt'))


if __name__ == '__main__':
    import configs

    c = configs.get_experiment_config_debug()
    print(nestedConfigs2dict(c))
