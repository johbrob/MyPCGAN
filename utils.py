from pathlib import Path
from enum import Enum
import local_vars
import scipy.io
import torch
import glob
import os

# ----------- Torch model related stuff -----------

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


def zero_grad(optimizers):
    [optimizer.zero_grad() for optimizer in optimizers.values()]


def step(optimizers):
    [optimizer.step() for optimizer in optimizers.values()]


def backward(losses):
    [loss['final'].backward() for loss in losses.values()]


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


# ----------- Saving Audio stuff -----------


def save_audio_file(file_path, sampling_rate, audio):
    if isinstance(file_path, str):
        file_path = Path(file_path)
    elif not isinstance(file_path, Path):
        raise ValueError('File path must be either str or Path')

    file_path.parent.mkdir(parents=True, exist_ok=True)
    audio = (audio.cpu().numpy() * 32768).astype("int16")
    scipy.io.wavfile.write(file_path, sampling_rate, audio)


def save_sample(save_dir, id, label, epoch, pred_label_male, pred_label_female, filtered_audio, audio_male,
                audio_female, original_audio, sampling_rate):
    speaker_digit_str = f'speaker_{id.item()}_digit_{label.item()}'
    speaker_digit_epoch_str = speaker_digit_str + f'_epoch_{epoch}'
    build_str = lambda gender, digit: speaker_digit_epoch_str + f'_sampled_gender_{gender}_predicted_digit_{digit}.wav'

    filtered_audio_file = os.path.join(save_dir, speaker_digit_epoch_str + '_filtered.wav')
    male_audio_file = os.path.join(save_dir, build_str('male', pred_label_male.item()))
    female_audio_file = os.path.join(save_dir, build_str('female', pred_label_female.item()))
    original_audio_file = os.path.join(save_dir, speaker_digit_str + '.wav')

    save_audio_file(filtered_audio_file, sampling_rate, filtered_audio.squeeze().detach().cpu())
    save_audio_file(male_audio_file, sampling_rate, audio_male.squeeze().detach().cpu())
    save_audio_file(female_audio_file, sampling_rate, audio_female.squeeze().detach().cpu())
    save_audio_file(original_audio_file, sampling_rate, original_audio.squeeze().detach().cpu())



# ----------- Path related stuff -----------


def get_run_dir(run_name):
    return local_vars.PWD + 'runs/audioMNIST/' + run_name


def create_run_subdir(run_name, sub_dir_name):
    path = os.path.join(get_run_dir(run_name), sub_dir_name)
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def create_subdir(dir, sub_dir_name):
    path = os.path.join(dir, sub_dir_name)
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


# ----------- Other stuff -----------


def nestedConfigs2dict(nested_config):
    return {k: v for attr in nested_config.__dict__.values() for k, v in vars(attr).items()}


def has_gradients(model, model_name):
    grads_are_zero = True
    max_abs_grad = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads_are_zero = False
            max_abs_grad = max(max_abs_grad, param.abs().max())

    if grads_are_zero:
        print(f'{model_name} has no gradients')
    else:
        print(f'{model_name} has gradients and largest is {max_abs_grad}')


if __name__ == '__main__':
    import configs
    c = configs.get_experiment_config_debug()
    print(nestedConfigs2dict(c))