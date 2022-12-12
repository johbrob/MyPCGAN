from DataManaging.AudioDatasets import AudioDataset
from nn.audio_mel import CustomAudioMelConverter, AudioMelConfig
from training.utils import preprocess_spectrograms
import torch
import local_vars
import utils


def main():
    device = 'cpu'

    train_data, test_data = AudioDataset.load()
    audio_mel_config = AudioMelConfig()
    audio_mel_converter = CustomAudioMelConverter(audio_mel_config)

    audio, secret, label, _, _ = test_data[0]

    spectrogram = audio_mel_converter.audio2mel(audio.unsqueeze(0)).detach()  # spectrogram: (bsz, n_mels, frames)
    spectrogram, means, stds = preprocess_spectrograms(spectrogram)
    spectrogram = spectrogram.unsqueeze(dim=1).to(device)  # spectrogram: (bsz, 1, n_mels, frames)

    unnormalized_spectrograms = torch.squeeze(spectrogram.to(device) * 3 * stds.to(device) + means.to(device))
    audio_again = audio_mel_converter.mel2audio(unnormalized_spectrograms.squeeze().detach().cpu())

    utils.save_audio_file(local_vars.PWD + 'tmp_original_audio.wav', 8000,audio.squeeze().detach().cpu())
    utils.save_audio_file(local_vars.PWD + 'recovered_audio.wav', 8000, audio_again.squeeze().detach().cpu())