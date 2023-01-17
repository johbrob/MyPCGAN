from neural_networks.whisper_encoder import WhisperSize


class AudioMelConfig:
    def __init__(self, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, center=False,
                 mel_fmin=0.0, mel_fmax=None, ngf=32, n_residual_layers=3, pretrained_path=False,
                 model_size=WhisperSize.BASE, sampling_rate=16000):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.center = center
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.sampling_rate = sampling_rate

        # for custom nn-mel2audio
        self.ngf = ngf
        self.n_residual_layers = n_residual_layers

        # for whisper
        self.model_size = model_size

        self.pretrained_path = pretrained_path
