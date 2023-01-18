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
from sanity_checks.audio2mel_comparisons.Whisper import from_wave
from sanity_checks.audio2mel_comparisons.Whisper import whisper_audio2mel
from sanity_checks.audio2mel_comparisons.LibRosa1 import librosa1_audio2mel


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

    print(np.amax(mel))
    print(np.amin(mel))
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
    mel = audio2mel(sample)

    if isinstance(mel, torch.Tensor):
        mel = mel.detach().cpu()

    if cutoff:
        mel = mel[..., :cutoff]

    audio = mel2audio(mel)

    if isinstance(audio, torch.Tensor):
        audio = audio.detach().squeeze()
    # if cutoff:
    #     audio = audio[..., :cutoff]
    utils.save_audio_file(path, sr, audio)


def get_samples(n_samples):
    train_data, test_data = CremaD.load(n_train_samples=10, n_test_samples=1)
    samples = [train_data[i][0] for i in range(n_samples)]
    samples = torch.stack(samples)
    return samples


def resample(samples, in_sr, out_sr):
    samples_new_sr = librosa.resample(samples.numpy(), orig_sr=in_sr, target_sr=out_sr)
    return torch.from_numpy(samples_new_sr)


def save_audio_and_spectrogram(sample, audio2mel, mel2audio, path, file_name, sr, hop_length, cutoff=None):
    save_audio(sample, audio2mel, mel2audio, os.path.join(path, file_name + '.wav'), sr=sr, cutoff=cutoff)
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
    save_audio_and_spectrogram(samples_22000[0], fft_22000, mel2audio, path, 'like_melgan_22000_fft_22000_0', 22000,
                               256)
    save_audio_and_spectrogram(samples_22000[1], fft_22000, mel2audio, path, 'like_melgan_22000_fft_22000_1', 22000,
                               256)
    save_audio_and_spectrogram(samples_22000[2], fft_22000, mel2audio, path, 'like_melgan_22000_fft_22000_2', 22000,
                               256)
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
    save_audio_and_spectrogram([samples[0]], whisper_audio2mel, mel2audio, path, 'whisper_16000_hop_length_160_0',
                               16000, 160)
    save_audio_and_spectrogram([samples[1]], whisper_audio2mel, mel2audio, path, 'whisper_16000_hop_length_160_1',
                               16000, 160)
    save_audio_and_spectrogram([samples[2]], whisper_audio2mel, mel2audio, path, 'whisper_16000_hop_length_160_2',
                               16000, 160)

    librosa_mel2audio = LibRosaMel2Audio(AudioMelConfig(hop_length=160))
    save_audio_and_spectrogram([samples[0]], whisper_audio2mel, librosa_mel2audio, path, 'whisper_librosa_0', 16000,
                               160, 300)
    save_audio_and_spectrogram([samples[1]], whisper_audio2mel, librosa_mel2audio, path, 'whisper_librosa_1', 16000,
                               160, 300)
    save_audio_and_spectrogram([samples[2]], whisper_audio2mel, librosa_mel2audio, path, 'whisper_librosa_2', 16000,
                               160, 300)

    save_audio_and_spectrogram([samples[0]], whisper_audio2mel, mel2audio, path, 'whisper_melgan_0', 16000, 160, 300)
    save_audio_and_spectrogram([samples[1]], whisper_audio2mel, mel2audio, path, 'whisper_melgan_1', 16000, 160, 300)
    save_audio_and_spectrogram([samples[2]], whisper_audio2mel, mel2audio, path, 'whisper_melgan_2', 16000, 160, 300)

    new_fft = Audio2Mel(n_mel_channels=80, sampling_rate=22000)
    librosa_mel2audio = LibRosaMel2Audio(AudioMelConfig())
    save_audio_and_spectrogram(samples[0], fft, librosa_mel2audio, path, 'melgan_librosa_0', 16000, 256)
    save_audio_and_spectrogram(samples[1], fft, librosa_mel2audio, path, 'melgan_librosa_1', 16000, 256)
    save_audio_and_spectrogram(samples[2], fft, librosa_mel2audio, path, 'melgan_librosa_2', 16000, 256)


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
    librosa2_audio2mel = MelGanAudio2Mel(AudioMelConfig())

    librosa_mel2audio = LibRosaMel2Audio(AudioMelConfig(n_fft=n_fft, hop_length=hop_length, win_length=30))

    melgan1 = MelGanMel2Audio(AudioMelConfig(sampling_rate=sampling_rate))
    melgan1.load_state_dict(torch.load(pretrained_path1, map_location=torch.device('cpu')))

    melgan2 = MelGanMel2Audio(AudioMelConfig(sampling_rate=sampling_rate))
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
    from sanity_checks.audio2mel_comparisons.Whisper import stft_fun as stft_whisper
    from sanity_checks.audio2mel_comparisons.LibRosa1 import stft_fun as stft_librosa1
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

    window = np.hanning(n_fft + 1)[:-1]
    frames = from_wave(sample, n_fft, hop_length, center)
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

    _, whisper_mels = whisper_audio2mel(sample, sampling_rate, n_fft, hop_length, center, n_mels)
    librosa1_mels = librosa1_audio2mel(sample, sampling_rate, n_fft, hop_length, center, n_mels)

    librosa2_audio2mel = MelGanAudio2Mel(
        AudioMelConfig(n_fft, hop_length, win_length, n_mels, center, mel_fmin, mel_fmax))
    _, librosa2_mels = librosa2_audio2mel(torch.from_numpy(sample))

    librosa2_mels = librosa2_mels.numpy()

    print(whisper_mels)
    print(librosa1_mels)
    print(librosa2_mels)

    # continue transform
    log_spec = np.log10(np.clip(whisper_mels, a_min=1e-10, a_max=None))
    max_log_spec = log_spec.max()
    log_spec_large = np.where(log_spec >= max_log_spec)
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    log_spec = log_spec * 4.0 - 4.0
    log_spec = 10 ** log_spec



def compare_stft():
    train_data, test_data = CremaD.load(n_train_samples=10, n_test_samples=1)
    sample = train_data[0][0]
    sample = sample.detach().numpy()

    n_fft = 1024
    hop_length = 256
    win_length = 1024
    n_mels = 80
    center = False
    mel_fmin = 0.0
    mel_fmax = None
    sampling_rate = 16000

    from librosa.filters import get_window
    from librosa.util import pad_center
    from librosa.util import expand_to
    fft_window = get_window('hann', win_length, fftbins=True)
    # Pad the window out to n_fft size
    fft_window = pad_center(fft_window, size=n_fft)
    # Reshape so that the window can be broadcast
    fft_window = expand_to(fft_window, ndim=1 + sample.ndim, axes=-2)
    fft_window = torch.from_numpy(fft_window).squeeze()
    fft_window = torch.hann_window(window_length=win_length)

    librosa_stft = librosa.stft(y=sample, n_fft=2048, hop_length=hop_length, win_length=win_length,
                                window="hann", center=False)
    librosa2_stft = librosa.spectrum.stft(y=sample, n_fft=2048, hop_length=hop_length, win_length=win_length,
                                window="hann", center=False)
    torch_stft = torch.stft(torch.from_numpy(sample), n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                     window=fft_window, center=False, return_complex=True).numpy()
    torch2_stft = torch.stft(torch.from_numpy(sample), n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                            window=fft_window, center=False, return_complex=True).numpy()


    fig = plt.figure(figsize=(24, 24))  # This has to be changed!!
    ax1 = fig.add_subplot(221)
    plt.imshow(librosa_stft, cmap=None, interpolation='nearest')
    plt.title('sample_spectrogram', fontsize=20)
    fig.savefig(path)
    plt.close(fig)
    print(torch_stft)


def whisper_decoding():
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
    whisper_mels = whisper_audio2mel(sample, sampling_rate, n_fft, hop_length, center, n_mels)
    mel2audio = LibRosaMel2Audio(AudioMelConfig(n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                                n_mels=n_mels, center=center))

    whisper_audio = mel2audio(whisper_mels)
    path = utils.create_run_subdir('just_a_test', '_i_say', 'whisper')
    utils.save_audio_file(path + 'audio.wav', sampling_rate, torch.from_numpy(whisper_audio))
    fig = plt.figure(figsize=(24, 24))  # This has to be changed!!
    ax1 = fig.add_subplot(221)
    p1 = librosa.display.specshow(whisper_mels, x_axis='time', y_axis='mel', sr=sampling_rate, fmax=4000,
                                  hop_length=hop_length,
                                  cmap='magma')
    plt.title('sample_spectrogram', fontsize=20)
    fig.savefig(path)
    plt.close(fig)


def compare_melgan_version():
    path = utils.create_run_subdir('just_a_test', 'melgan_comparisons', 'audio')
    pretrained_path1 = 'neural_networks/pretrained_weights/multi_speaker.pt'
    a2m1 = Audio2Mel(n_mel_channels=80, sampling_rate=22050)
    m2a1 = Generator(80, 32, 3)
    m2a1.load_state_dict(torch.load(pretrained_path1, map_location=torch.device('cpu')))

    a2m2 = MelGanAudio2Mel(AudioMelConfig(sampling_rate=22050))
    m2a2 = MelGanMel2Audio(AudioMelConfig(sampling_rate=22050))
    m2a2.load_state_dict(torch.load(pretrained_path1, map_location=torch.device('cpu')))

    samples = get_samples(3)

    # save originals
    utils.save_audio_file(os.path.join(path, 'original_0' + '.wav'), 16000, samples[0].detach().squeeze())
    utils.save_audio_file(os.path.join(path, 'original_1' + '.wav'), 16000, samples[1].detach().squeeze())
    utils.save_audio_file(os.path.join(path, 'original_2' + '.wav'), 16000, samples[2].detach().squeeze())

    # save 22050 with fft(22050) how it should be
    samples_22050 = resample(samples, 16000, 22050)
    save_audio_and_spectrogram(samples_22050[0], a2m1, m2a1, path, 'melgan1_22050_0', 22050, 256)
    save_audio_and_spectrogram(samples_22050[1], a2m1, m2a1, path, 'melgan1_22050_1', 22050, 256)
    save_audio_and_spectrogram(samples_22050[2], a2m1, m2a1, path, 'melgan1_22050_2', 22050, 256)
    save_audio_and_spectrogram(samples_22050[0], a2m2, m2a2, path, 'melgan2_22050_0', 22050, 256)
    save_audio_and_spectrogram(samples_22050[1], a2m2, m2a2, path, 'melgan2_22050_1', 22050, 256)
    save_audio_and_spectrogram(samples_22050[2], a2m2, m2a2, path, 'melgan2_22050_2', 22050, 256)

    save_audio_and_spectrogram(samples[0], a2m1, m2a1, path, 'melgan1_16000_0', 16000, 256)
    save_audio_and_spectrogram(samples[1], a2m1, m2a1, path, 'melgan1_16000_1', 16000, 256)
    save_audio_and_spectrogram(samples[2], a2m1, m2a1, path, 'melgan1_16000_2', 16000, 256)
    save_audio_and_spectrogram(samples[0], a2m2, m2a2, path, 'melgan2_16000_0', 16000, 256)
    save_audio_and_spectrogram(samples[1], a2m2, m2a2, path, 'melgan2_16000_1', 16000, 256)
    save_audio_and_spectrogram(samples[2], a2m2, m2a2, path, 'melgan2_16000_2', 16000, 256)


def whisper_as_melgan():
    path = utils.create_run_subdir('just_a_test', 'whisper_as_melgan', 'audio')
    pretrained_path1 = 'neural_networks/pretrained_weights/multi_speaker.pt'
    config = AudioMelConfig(sampling_rate=16000, hop_length=160)
    a2m_melgan = MelGanAudio2Mel(config)
    m2a = MelGanMel2Audio(config)
    m2a.load_state_dict(torch.load(pretrained_path1, map_location=torch.device('cpu')))
    a2m_whisper = CustomWhisperAudio2Mel(config)

    samples = get_samples(3)

    melgan_mel = a2m_melgan(samples[0])
    whisper_mel = a2m_whisper(samples[0])

    print('melgan', torch.amax(melgan_mel), torch.amin(melgan_mel))
    print('whisper', torch.amax(whisper_mel), torch.amin(whisper_mel))

    def save(mel, sr, path, m2a):
        if isinstance(mel, torch.Tensor):
            mel = mel.detach().cpu()
        audio = m2a(mel)
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().squeeze()
        utils.save_audio_file(path, sr, audio)


    # # save originals
    # utils.save_audio_file(os.path.join(path, 'original_0' + '.wav'), 16000, samples[0].detach().squeeze())
    # utils.save_audio_file(os.path.join(path, 'original_1' + '.wav'), 16000, samples[1].detach().squeeze())
    # utils.save_audio_file(os.path.join(path, 'original_2' + '.wav'), 16000, samples[2].detach().squeeze())
    #
    # save_audio_and_spectrogram(samples[0], a2m_melgan, m2a, path, 'melgan_0', 16000, 160)
    # save_audio_and_spectrogram(samples[1], a2m_melgan, m2a, path, 'melgan_1', 16000, 160)
    # save_audio_and_spectrogram(samples[2], a2m_melgan, m2a, path, 'melgan_2', 16000, 160)
    #
    # save_audio_and_spectrogram(samples[0, :], a2m_whisper, m2a, path, 'whisper_0', 16000, 160)
    # save_audio_and_spectrogram(samples[1, :], a2m_whisper, m2a, path, 'whisper_1', 16000, 160)
    # save_audio_and_spectrogram(samples[2, :], a2m_whisper, m2a, path, 'whisper_2', 16000, 160)
    #
    # save_audio_and_spectrogram(samples[0, :], a2m_whisper, m2a, path, 'whisper_22050_256_0', 22050, 256)
    # save_audio_and_spectrogram(samples[1, :], a2m_whisper, m2a, path, 'whisper_22050_2561', 22050, 256)
    # save_audio_and_spectrogram(samples[2, :], a2m_whisper, m2a, path, 'whisper_22050_256_2', 22050, 256)
    #
    # a2m_melgan2 = MelGanAudio2Mel(AudioMelConfig(sampling_rate=22050, hop_length=256))
    # samples_22050 = resample(samples, 16000, 22050)
    # save_audio_and_spectrogram(samples_22050[0], a2m_melgan2, m2a, path, 'melgan_22050_256_0', 22050, 256)
    # save_audio_and_spectrogram(samples_22050[1], a2m_melgan2, m2a, path, 'melgan_22050_256_1', 22050, 256)
    # save_audio_and_spectrogram(samples_22050[2], a2m_melgan2, m2a, path, 'melgan_22050_256_2', 22050, 256)



if __name__ == '__main__':
    # basic_test()
    # main()
    # like_melgan()
    # limits_of_melgan()
    stft()
    # make_mels()
    # whisper_decoding()
    # compare_stft()
    compare_melgan_version()