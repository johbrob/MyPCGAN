from audio_mel_conversion import AudioMelConfig, CustomMel2Audio, LibRosaMel2Audio, LibRosaAudio2Mel, \
    LibRosaAudio2Mel2
# from audio_mel_conversion import WhisperAudio2Mel
from neural_networks.whisper_encoder import WhisperSize
from datasets import CremaD
import utils
import torch
import matplotlib.pyplot as plt
import os
import librosa.display
from tmp_models import Audio2Mel, Generator
from transformers import WhisperFeatureExtractor


class WhisperAudio2Mel:
    def __init__(self, config, chunk_length):
        self.feature_extractor = WhisperFeatureExtractor(config.n_mels, config.sampling_rate, config.hop_length,
                                                         chunk_length, config.n_fft)
        self.sampling_rate = config.sampling_rate

    def __call__(self, data):
        data = [audio.squeeze().cpu().numpy() for audio in data]
        return self.feature_extractor(data, return_tensors="pt", sampling_rate=self.sampling_rate,
                                      padding=False).input_features


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


def main():
    feature_size = 80
    sampling_rate = 16000
    hop_length = 160
    chunk_length = 30
    n_fft = 400
    padding_value = 0.0

    path = utils.create_run_subdir('just_a_test', '_i_say', 'audio')
    whisper_audio2mel = WhisperAudio2Mel(AudioMelConfig(WhisperSize.TINY))
    librosa_audio2mel = LibRosaAudio2Mel(AudioMelConfig())
    librosa2_audio2mel = LibRosaAudio2Mel2(AudioMelConfig())

    librosa_mel2audio = LibRosaMel2Audio(AudioMelConfig(n_fft=n_fft, hop_length=hop_length, win_length=30))

    melgan1 = CustomMel2Audio(AudioMelConfig(sampling_rate=sampling_rate))
    pretrained_path1 = 'neural_networks/pretrained_weights/multi_speaker.pt'
    melgan1.load_state_dict(torch.load(pretrained_path1, map_location=torch.device('cpu')))

    melgan2 = CustomMel2Audio(AudioMelConfig(sampling_rate=sampling_rate))
    pretrained_path2 = 'neural_networks/pretrained_weights/best_netG_epoch_2120.pt'
    melgan2.load_state_dict(torch.load(pretrained_path2, map_location=torch.device('cpu')))

    fft = Audio2Mel(n_mel_channels=80, sampling_rate=16000)
    Mel2Audio = Generator(80, 32, 3)
    Mel2Audio.load_state_dict(torch.load(pretrained_path1, map_location=torch.device('cpu')))

    train_data, test_data = CremaD.load(n_train_samples=10, n_test_samples=1)
    sample = train_data[0][0]
    sample = sample.unsqueeze(dim=0)
    sample_22500 = librosa.resample(sample.numpy(), orig_sr=16000, target_sr=sampling_rate)
    sample = torch.from_numpy(sample_22500)

    # save_audio(sample, fft, Mel2Audio, os.path.join(path, 'from_edvin_audio.wav'))
    # save_audio(sample, whisper_audio2mel, Mel2Audio, os.path.join(path, 'whisper_edvin_audio.wav'), cutoff=300)
    # save_audio(sample, whisper_audio2mel, melgan2, os.path.join(path, 'whisper_melgan2_audio.wav'), cutoff=300)
    # save_audio(sample, librosa_audio2mel, melgan1, os.path.join(path, 'librosa_melgan1_audio.wav'))
    # save_audio(sample, librosa_audio2mel, melgan2, os.path.join(path, 'librosa_melgan2_audio.wav'))
    # save_audio(sample, librosa2_audio2mel, melgan1, os.path.join(path, 'librosa2_melgan1_audio.wav'))
    # save_audio(sample, librosa2_audio2mel, melgan2, os.path.join(path, 'librosa2_melgan2_audio.wav'))
    # save_audio(sample, whisper_audio2mel, librosa_mel2audio, os.path.join(path, 'whisper_librosa_audio.wav'), cutoff=300)
    # save_audio(sample, librosa_audio2mel, librosa_mel2audio, os.path.join(path, 'librosa_librosa_audio.wav'))
    # save_audio(sample, librosa2_audio2mel, librosa_mel2audio, os.path.join(path, 'librosa2_librosa_audio.wav'))

    save_mel_image(sample, whisper_audio2mel, os.path.join(path, 'whisper_spectrogram.png'), cutoff=300)
    save_mel_image(sample, librosa_audio2mel, os.path.join(path, 'librosa_spectrogram.png'))
    save_mel_image(sample, librosa2_audio2mel, os.path.join(path, 'librosa2_spectrogram.png'))


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


if __name__ == '__main__':
    basic_test()
