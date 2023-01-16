import numpy as np

from audio_mel_conversion import AudioMelConfig, CustomMel2Audio, LibRosaMel2Audio, LibRosaAudio2Mel, \
    LibRosaAudio2Mel2
from audio_mel_conversion import WhisperAudio2Mel
from neural_networks.whisper_encoder import WhisperSize
from datasets import CremaD, AudioMNIST
import utils
import torch
import matplotlib.pyplot as plt
import os
import librosa.display
from tmp_models import Audio2Mel, Generator
from transformers import WhisperFeatureExtractor


# class WhisperAudio2Mel:
#     def __init__(self, config, chunk_length):
#         self.feature_extractor = WhisperFeatureExtractor(config.n_mels, config.sampling_rate, config.hop_length,
#                                                          chunk_length, config.n_fft)
#         self.sampling_rate = config.sampling_rate
#
#     def __call__(self, data):
#         data = [audio.squeeze().cpu().numpy() for audio in data]
#         return self.feature_extractor(data, return_tensors="pt", sampling_rate=self.sampling_rate,
#                                       padding=False).input_features


def save_mel_image(sample, audio2mel, path, sr, hop_length, cutoff=None):
    mel = audio2mel(sample).detach()
    mel = mel.squeeze().cpu().numpy()

    if cutoff:
        mel = mel[..., :cutoff]

    print(mel.shape)
    fig = plt.figure(figsize=(24, 24))  # This has to be changed!!
    ax1 = fig.add_subplot(221)
    p1 = librosa.display.specshow(mel, x_axis='time', y_axis='mel', sr=sr, fmax=4000, hop_length=hop_length,
                                  cmap='magma')
    plt.title('sample_spectrogram', fontsize=20)
    fig.savefig(path)
    plt.close(fig)


def save_audio(sample, audio2mel, mel2audio, path, sr, cutoff=None):
    mel = audio2mel(sample).detach()

    if cutoff:
        mel = mel[..., :cutoff]

    audio = mel2audio(mel.squeeze().cpu())

    # if cutoff:
    #     audio = audio[..., :cutoff]
    utils.save_audio_file(path, sr, audio.detach().squeeze())


def get_samples(n_samples):
    train_data, test_data = CremaD.load(n_train_samples=10, n_test_samples=1)
    samples = [train_data[i][0] for i in range(n_samples)]
    samples = torch.stack(samples)
    return samples


def resample(samples, in_sr, out_sr):
    samples_new_sr = librosa.resample(samples.numpy(), orig_sr=in_sr, target_sr=out_sr)
    return torch.from_numpy(samples_new_sr)


def save_audio_and_spectrogram(sample, audio2mel, mel2audio, path, file_name, sr, hop_length):
    save_audio(sample, audio2mel, mel2audio, os.path.join(path, file_name + '.wav'), sr=sr)
    save_mel_image(sample, audio2mel, os.path.join(path, file_name + '.png'), sr, hop_length)

def convert_and_save(audio2mel, mel2audio, in_sr, out_sr, path, file_name, hop_length):
    sample = get_samples(in_sr, out_sr, 1)
    if in_sr != out_sr:
        sample = resample(sample, in_sr, out_sr)
    save_audio_and_spectrogram(sample, audio2mel, mel2audio, path, file_name, out_sr, hop_length)

def like_melgan():
    pretrained_path1 = 'neural_networks/pretrained_weights/multi_speaker.pt'
    fft = Audio2Mel(n_mel_channels=80, sampling_rate=22050)
    mel2audio = Generator(80, 32, 3)
    mel2audio.load_state_dict(torch.load(pretrained_path1, map_location=torch.device('cpu')))

    path = utils.create_run_subdir('just_a_test', '_i_say', 'audio')
    convert_and_save(fft, mel2audio, 16000, 22050, path, 'like_melgan', 256)


def limits_of_melgan():
    path = utils.create_run_subdir('just_a_test', 'melgan', 'audio')
    pretrained_path1 = 'neural_networks/pretrained_weights/multi_speaker.pt'
    fft = Audio2Mel(n_mel_channels=80, sampling_rate=22050)
    mel2audio = Generator(80, 32, 3)
    mel2audio.load_state_dict(torch.load(pretrained_path1, map_location=torch.device('cpu')))

    whisper_audio2mel = WhisperAudio2Mel(AudioMelConfig(model_size=WhisperSize.TINY))

    samples = get_samples(3)

    # save originals
    utils.save_audio_file(os.path.join(path, 'original_0' + '.wav'), 16000, samples[0].detach().squeeze())
    utils.save_audio_file(os.path.join(path, 'original_1' + '.wav'), 16000, samples[1].detach().squeeze())
    utils.save_audio_file(os.path.join(path, 'original_2' + '.wav'), 16000, samples[2].detach().squeeze())

    # save 22050 with fft(22050) how it should be
    samples_22050 = resample(samples, 16000, 22050)
    save_audio_and_spectrogram(samples_22050[0], fft, mel2audio, path, 'like_melgan_22050_0', 22050, 256)
    save_audio_and_spectrogram(samples_22050[1], fft, mel2audio, path, 'like_melgan_22050_1', 22050, 256)
    save_audio_and_spectrogram(samples_22050[2], fft, mel2audio, path, 'like_melgan_22050_2', 22050, 256)
    # save audio with sample rate slightly too low but still fft(22050)
    samples_22000 = resample(samples, 16000, 22000)
    save_audio_and_spectrogram(samples_22000[0], fft, mel2audio, path, 'like_melgan_22000_0', 22000, 256)
    save_audio_and_spectrogram(samples_22000[1], fft, mel2audio, path, 'like_melgan_22000_1', 22000, 256)
    save_audio_and_spectrogram(samples_22000[2], fft, mel2audio, path, 'like_melgan_22000_2', 22000, 256)
    # save audio with sample rate slightly too high but still fft(22050)
    samples_22100 = resample(samples, 16000, 22100)
    save_audio_and_spectrogram(samples_22100[0], fft, mel2audio, path, 'like_melgan_22100_0', 22100, 256)
    save_audio_and_spectrogram(samples_22100[1], fft, mel2audio, path, 'like_melgan_22100_1', 22100, 256)
    save_audio_and_spectrogram(samples_22100[2], fft, mel2audio, path, 'like_melgan_22100_2', 22100, 256)
    # save audio with original sample rate but still fft(22050)
    save_audio_and_spectrogram(samples[0], fft, mel2audio, path, 'like_melgan_16000_0', 16000, 256)
    save_audio_and_spectrogram(samples[1], fft, mel2audio, path, 'like_melgan_16000_1', 16000, 256)
    save_audio_and_spectrogram(samples[2], fft, mel2audio, path, 'like_melgan_16000_2', 16000, 256)
    # save audio with original sample rate and still fft(16000)
    fft_16000 = Audio2Mel(n_mel_channels=80, sampling_rate=16000)
    save_audio_and_spectrogram(samples[0], fft_16000, mel2audio, path, 'like_melgan_16000_fft_16000_0', 16000, 256)
    save_audio_and_spectrogram(samples[1], fft_16000, mel2audio, path, 'like_melgan_16000_fft_16000_1', 16000, 256)
    save_audio_and_spectrogram(samples[2], fft_16000, mel2audio, path, 'like_melgan_16000_fft_16000_2', 16000, 256)
    # save audio with slightly too low sample rate and matching fft(22000)
    fft_22000 = Audio2Mel(n_mel_channels=80, sampling_rate=22000)
    save_audio_and_spectrogram(samples_22000[0], fft_22000, mel2audio, path, 'like_melgan_22000_fft_22000_0', 22000, 256)
    save_audio_and_spectrogram(samples_22000[1], fft_22000, mel2audio, path, 'like_melgan_22000_fft_22000_1', 22000, 256)
    save_audio_and_spectrogram(samples_22000[2], fft_22000, mel2audio, path, 'like_melgan_22000_fft_22000_2', 22000, 256)
    # save 22050 with fft(22050) with too low hop_length
    save_audio_and_spectrogram(samples_22050[0], fft, mel2audio, path, 'like_melgan_22050_200_0', 22050, 200)
    save_audio_and_spectrogram(samples_22050[1], fft, mel2audio, path, 'like_melgan_22050_200_1', 22050, 200)
    save_audio_and_spectrogram(samples_22050[2], fft, mel2audio, path, 'like_melgan_22050_200_2', 22050, 200)
    # save 22050 with fft(22050) with too high hop_length
    samples_22050 = resample(samples, 16000, 22050)
    save_audio_and_spectrogram(samples_22050[0], fft, mel2audio, path, 'like_melgan_22050_300_0', 22050, 300)
    save_audio_and_spectrogram(samples_22050[1], fft, mel2audio, path, 'like_melgan_22050_300_1', 22050, 300)
    save_audio_and_spectrogram(samples_22050[2], fft, mel2audio, path, 'like_melgan_22050_300_2', 22050, 300)
    # save audio using whisper audio2mel with original sample rate
    save_audio_and_spectrogram([samples[0]], whisper_audio2mel, mel2audio, path, 'whisper_16000_0', 16000, 256)
    save_audio_and_spectrogram([samples[1]], whisper_audio2mel, mel2audio, path, 'whisper_16000_1', 16000, 256)
    save_audio_and_spectrogram([samples[2]], whisper_audio2mel, mel2audio, path, 'whisper_16000_2', 16000, 256)
    # save audio using whisper audio2mel with original sample rate and hop length = 160
    save_audio_and_spectrogram([samples[0]], whisper_audio2mel, mel2audio, path, 'whisper_16000_hop_length_160_0', 16000, 160)
    save_audio_and_spectrogram([samples[1]], whisper_audio2mel, mel2audio, path, 'whisper_16000_hop_length_160_1', 16000, 160)
    save_audio_and_spectrogram([samples[2]], whisper_audio2mel, mel2audio, path, 'whisper_16000_hop_length_160_2', 16000, 160)

def main():
    feature_size = 80
    sampling_rate = 16000
    hop_length = 160
    chunk_length = 30
    n_fft = 400
    padding_value = 0.0

    pretrained_path1 = 'neural_networks/pretrained_weights/multi_speaker.pt'
    pretrained_path2 = 'neural_networks/pretrained_weights/best_netG_epoch_2120.pt'
    pretrained_path3 = 'neural_networks/pretrained_weights/linda_johnson.pt'

    path = utils.create_run_subdir('just_a_test', '_i_say', 'audio')
    # whisper_audio2mel = WhisperAudio2Mel(AudioMelConfig(WhisperSize=WhisperSize.TINY), chunk_length=chunk_length)
    whisper_audio2mel = WhisperAudio2Mel(AudioMelConfig(model_size=WhisperSize.TINY))
    librosa_audio2mel = LibRosaAudio2Mel(AudioMelConfig())
    librosa2_audio2mel = LibRosaAudio2Mel2(AudioMelConfig())

    librosa_mel2audio = LibRosaMel2Audio(AudioMelConfig(n_fft=n_fft, hop_length=hop_length, win_length=30))

    melgan1 = CustomMel2Audio(AudioMelConfig(sampling_rate=sampling_rate))
    melgan1.load_state_dict(torch.load(pretrained_path1, map_location=torch.device('cpu')))

    melgan2 = CustomMel2Audio(AudioMelConfig(sampling_rate=sampling_rate))
    melgan2.load_state_dict(torch.load(pretrained_path2, map_location=torch.device('cpu')))

    fft = Audio2Mel(n_mel_channels=80, sampling_rate=16000)
    Mel2Audio = Generator(80, 32, 3)
    Mel2Audio.load_state_dict(torch.load(pretrained_path1, map_location=torch.device('cpu')))

    Mel2Audio2 = Generator(80, 32, 3)
    Mel2Audio2.load_state_dict(torch.load(pretrained_path2, map_location=torch.device('cpu')))

    Mel2Audio3 = Generator(80, 32, 3)
    Mel2Audio3.load_state_dict(torch.load(pretrained_path3, map_location=torch.device('cpu')))


def basic_test():
    # save path
    path = utils.create_run_subdir('just_a_test', 'basic_test', 'audio')

    # configs
    standard_config = AudioMelConfig(n_fft=1024, hop_length=256, win_length=1024, n_mels=80, center=False, mel_fmin=0.0,
                                     mel_fmax=None, sampling_rate=16000)
    hz_22500_config = AudioMelConfig(n_fft=1024, hop_length=256, win_length=1024, n_mels=80, center=False, mel_fmin=0.0,
                                     mel_fmax=None, sampling_rate=22500)
    win_length_30_config = AudioMelConfig(n_fft=1024, hop_length=256, win_length=30, n_mels=80, center=False,
                                          mel_fmin=0.0, mel_fmax=None, sampling_rate=16000)

    train_data, test_data = CremaD.load(n_train_samples=10, n_test_samples=1)
    sample = train_data[0][0]
    sample = sample.unsqueeze(dim=0)

    # converters
    converters = {
        'lr_lr_standard': {'a2m': LibRosaAudio2Mel(standard_config),
                           'm2a': LibRosaMel2Audio(standard_config),
                           'sample': sample, 'sr': 16000},
        'lr_lr_hz_22500': {'a2m': LibRosaAudio2Mel(hz_22500_config),
                           'm2a': LibRosaMel2Audio(hz_22500_config),
                           'sample': torch.from_numpy(
                               librosa.resample(sample.numpy(), orig_sr=16000, target_sr=22500)),
                           'sr': 22500},
        'lr_lr_win_length_30': {'a2m': LibRosaAudio2Mel(win_length_30_config),
                                'm2a': LibRosaMel2Audio(win_length_30_config),
                                'sample': sample, 'sr': 16000},
        'w_lr_standard_chunk_length_30': {'a2m': WhisperAudio2Mel(standard_config, chunk_length=30),
                                          'm2a': LibRosaMel2Audio(standard_config),
                                          'sample': sample, 'sr': 16000},
        'w_lr_standard_chunk_length_30_win_length_30': {'a2m': WhisperAudio2Mel(standard_config, chunk_length=30),
                                                        'm2a': LibRosaMel2Audio(win_length_30_config),
                                                        'sample': sample, 'sr': 16000},
        'w_lr_standard_chunk_length_1024': {'a2m': WhisperAudio2Mel(standard_config, chunk_length=1024),
                                            'm2a': LibRosaMel2Audio(standard_config),
                                            'sample': sample, 'sr': 16000}

    }

    for name, cv in converters.items():
        save_audio(cv['sample'], cv['a2m'], cv['m2a'], os.path.join(path, f'{name}_audio.wav'), cv['sr'])
        save_mel_image(cv['sample'], cv['a2m'], os.path.join(path, f'{name}.png'), cv['sr'], hop_length=256)


def stft():
    n_fft = 1024
    hop_length = 256
    win_length = 1024
    n_mels = 80
    center = False
    mel_fmin = 0.0
    mel_fmax = None
    sampling_rate = 16000

    train_data, test_data = CremaD.load(n_train_samples=10, n_test_samples=1)
    sample = train_data[0][0]
    sample = sample.detach().numpy()
    # sample = sample.unsqueeze(dim=0)

    import numpy as np
    from audio2mel_comparisons.Whisper import stft_fun as stft_whisper
    from audio2mel_comparisons.Whisper import fram_wave
    from audio2mel_comparisons.LibRosa1 import stft_fun as stft_librosa1

    window = np.hanning(n_fft + 1)[:-1]
    frames = fram_wave(sample, n_fft, hop_length, center)
    stft_whisper = stft_whisper(frames, window=window, n_fft=n_fft)
    magnitudes = np.abs(stft_whisper[:, :-1]) ** 2

    power = 2.0
    window = 'hann'
    pad_mode = 'constant'
    # Otherwise, compute a magnitude spectrogram from input
    stft_librosa1 = stft_librosa1(sample, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=center,
                                  window=window, pad_mode=pad_mode)
    S = (np.abs(stft_librosa1) ** power)

    # stft are the same!

def make_mels():
    import numpy as np
    from audio2mel_comparisons.Whisper import whisper_audio2mel
    from audio2mel_comparisons.LibRosa1 import librosa1_audio2mel

    n_fft = 1024
    hop_length = 256
    win_length = 1024
    n_mels = 80
    center = False
    mel_fmin = 0.0
    mel_fmax = None
    sampling_rate = 16000

    print('whatttttututop')

    train_data, test_data = CremaD.load(n_train_samples=10, n_test_samples=1)
    sample = train_data[0][0]
    sample = sample.detach().numpy()

    whisper_mels = whisper_audio2mel(sample, sampling_rate, n_fft, hop_length, center, n_mels)
    librosa1_mels = librosa1_audio2mel(sample, sampling_rate, n_fft, hop_length, center, n_mels)

    librosa2_audio2mel = LibRosaAudio2Mel2(AudioMelConfig(n_fft, hop_length, win_length, n_mels, center, mel_fmin, mel_fmax))
    librosa2_mels = librosa2_audio2mel(torch.from_numpy(sample))
    print(whisper_mels)
    print(librosa1_mels)


if __name__ == '__main__':
    # basic_test()
    # main()
    # like_melgan()
    # limits_of_melgan()
    # stft()
    make_mels()
