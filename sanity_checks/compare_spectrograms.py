from audio_mel_conversion import AudioMelConfig, LibRosaMel2Audio, MelGanMel2Audio, LibRosaAudio2Mel, \
    MelGanAudio2Mel, CustomWhisperAudio2Mel, WhisperAudio2Mel
from neural_networks.whisper_encoder import WhisperSize
from datasets import CremaD
import utils
import torch
import matplotlib.pyplot as plt
import os
import librosa.display
from sanity_checks.tmp_models import Audio2Mel, Generator
import numpy as np
from sanity_checks.audio2mel_comparisons.Whisper import from_wave, whisper_audio2mel
from sanity_checks.audio2mel_comparisons.Whisper import get_components as whisper_components
from sanity_checks.audio2mel_comparisons.LibRosa1 import librosa1_audio2mel
from sanity_checks.audio2mel_comparisons.MelGan import get_components as melgan_components


def get_samples(n_samples):
    train_data, test_data = CremaD.load(n_train_samples=10, n_test_samples=1)
    samples = [train_data[i][0] for i in range(n_samples)]
    samples = torch.stack(samples)
    return samples


def main():
    path = utils.create_run_subdir('just_a_test', 'whisper_as_melgan', 'audio')
    pretrained_path1 = 'neural_networks/pretrained_weights/multi_speaker.pt'
    config = AudioMelConfig(sampling_rate=16000, hop_length=160)
    a2m_melgan = MelGanAudio2Mel(config)
    m2a = MelGanMel2Audio(config)
    m2a.load_state_dict(torch.load(pretrained_path1, map_location=torch.device('cpu')))
    a2m_whisper = CustomWhisperAudio2Mel(config)

    samples = get_samples(3)

    window_w, frames_w, stft_w, magnitudes_w, filters_w, mel_spec_w = whisper_components(samples[0], 1024, 16000, 80, 256, False)
    window_m, frames_m, stft_m, magnitudes_m, filters_m, mel_spec_m = melgan_components(samples[0], 16000, 1024, 256, False, 80, 1024)

    window_m = window_m.numpy()
    stft_m = stft_m.squeeze().numpy()
    magnitudes_m = magnitudes_m.squeeze().numpy()
    filters_m = filters_m.numpy()
    mel_spec_m = mel_spec_m.squeeze().numpy()

    mel_spec_w = mel_spec_w[:, :mel_spec_m.shape[-1]]
    stft_w = stft_w[:, :stft_m.shape[-1]]
    magnitudes_w = magnitudes_w[:, :magnitudes_m.shape[-1]]

    window_diff = window_m - window_w
    stft_diff = stft_m - stft_w
    magnitudes_diff = magnitudes_m - magnitudes_w
    filters_diff = filters_m - filters_w
    mel_spec_diff = mel_spec_m - mel_spec_w
    print('hej')


    # melgan_mel = a2m_melgan(samples[0])
    # whisper_mel = a2m_whisper(samples[0])
    #
    # print('melgan', torch.amax(melgan_mel), torch.amin(melgan_mel))
    # print('whisper', torch.amax(whisper_mel), torch.amin(whisper_mel))