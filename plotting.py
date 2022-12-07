import matplotlib.pyplot as plt
import librosa.display
from pathlib import Path
import torch
import os


def comparison_plot_filtergen(file_path, orig_spec, male_spec, female_spec, orig_title, male_title, female_title):
    orig_spec_np = torch.squeeze(orig_spec).cpu().numpy()
    male_spec_np = torch.squeeze(male_spec).cpu().numpy()
    female_spec_np = torch.squeeze(female_spec).cpu().numpy()
    fig = plt.figure(figsize=(30, 8))
    ax1 = fig.add_subplot(131)
    p1 = librosa.display.specshow(orig_spec_np, x_axis='time', y_axis='mel', sr=8000, fmax=4000, hop_length=256)
    plt.title(orig_title, fontsize=18)
    ax2 = fig.add_subplot(132)
    p2 = librosa.display.specshow(male_spec_np, x_axis='time', y_axis='mel', sr=8000, fmax=4000, hop_length=256)
    plt.title(male_title, fontsize=18)
    ax3 = fig.add_subplot(133)
    p3 = librosa.display.specshow(female_spec_np, x_axis='time', y_axis='mel', sr=8000, fmax=4000, hop_length=256)
    plt.title(female_title, fontsize=18)
    fig.savefig(file_path)
    plt.close(fig)


def comparison_plot_pcgan(original_spectrogram, filtered_spectrogram, male_spectrogram, female_spectrogram,
                          secret, label, pred_secret_male, pred_secret_female, pred_label_male, pred_label_female,
                          male_distortion, female_distortion, sample_distortion, example_dirs, epoch, id):

    pred_secret_male = 'male' if pred_secret_male else 'female'
    pred_secret_female = 'female' if pred_secret_female else 'male'
    gender_title = 'male' if secret else 'female'

    orig_title = f'Original spectrogram - Gender: {gender_title} - Digit: {label.item()}'
    filtered_title = 'Filtered spectrogram'
    male_title = 'Sampled/predicted gender: male / {} | Predicted digit: {} \n Distortion loss: {:5.5f} (original) | {:5.5f} (female) ({}_loss)'.format(
        pred_secret_male, pred_label_male.item(), male_distortion, sample_distortion, 'l1')
    female_title = 'Sampled/predicted gender: female / {} | Predicted digit: {} \n Distortion loss: {:5.5f} (original) | {:5.5f} (male) ({}_loss)'.format(
        pred_secret_female, pred_label_female.item(), female_distortion, sample_distortion,
        'l1')

    speaker_digit_str = f'speaker_{id.item()}_digit_{label.item()}'
    speaker_digit_epoch_str = speaker_digit_str + f'_epoch_{epoch}'
    file_path = os.path.join(example_dirs['spec'], speaker_digit_epoch_str)
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    orig_spec_np = torch.squeeze(original_spectrogram).cpu().numpy()
    filtered_spec_np = torch.squeeze(filtered_spectrogram).cpu().numpy()
    male_spec_np = torch.squeeze(male_spectrogram).cpu().numpy()
    female_spec_np = torch.squeeze(female_spectrogram).cpu().numpy()
    fig = plt.figure(figsize=(24, 24))  # This has to be changed!!
    ax1 = fig.add_subplot(221)
    p1 = librosa.display.specshow(orig_spec_np, x_axis='time', y_axis='mel', sr=8000, fmax=4000, hop_length=256,
                                  cmap='magma')
    plt.title(orig_title, fontsize=20)
    ax2 = fig.add_subplot(222)
    p2 = librosa.display.specshow(filtered_spec_np, x_axis='time', y_axis='mel', sr=8000, fmax=4000, hop_length=256,
                                  cmap='magma')
    plt.title(filtered_title, fontsize=20)
    ax3 = fig.add_subplot(223)
    p3 = librosa.display.specshow(male_spec_np, x_axis='time', y_axis='mel', sr=8000, fmax=4000, hop_length=256,
                                  cmap='magma')
    plt.title(male_title, fontsize=20)
    ax4 = fig.add_subplot(224)
    p4 = librosa.display.specshow(female_spec_np, x_axis='time', y_axis='mel', sr=8000, fmax=4000, hop_length=256,
                                  cmap='magma')
    plt.title(female_title, fontsize=20)
    fig.savefig(file_path)
    plt.close(fig)
