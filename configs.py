from loss_compiling import LossComputeConfig
import numpy as np


class TrainingConfig:
    def __init__(self, run_name, train_batch_size, test_batch_size, train_num_workers, test_num_workers, save_interval,
                 checkpoint_interval, updates_per_evaluation, gradient_accumulation, lr, epochs, n_samples):
        self.run_name = run_name + '_' + self.random_id()
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_num_workers = train_num_workers
        self.test_num_workers = test_num_workers
        self.save_interval = save_interval
        self.checkpoint_interval = checkpoint_interval
        self.updates_per_evaluation = updates_per_evaluation
        self.gradient_accumulation = gradient_accumulation
        self.lr = lr
        self.epochs = epochs
        self.n_samples = n_samples

    def random_id(self):
        return str(np.random.randint(0, 9, 7))[1:-1].replace(' ', '')


class Audio2MelConfig:
    def __init__(self, n_fft=1024, hop_length=256, win_length=1024, sampling_rate=22050, n_mels=80,
                 mel_fmin=0.0, mel_fmax=None):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mels = n_mels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax


class Mel2AudioConfig:
    def __init__(self, input_size, ngf, n_residual_layers):
        self.input_size = input_size
        self.ngf = ngf
        self.n_residual_layers = n_residual_layers


class UnetConfig:
    def __init__(self, kernel_size, embedding_dim, noise_dim):
        self.kernel_size = kernel_size
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim


class ExperimentConfig:
    def __init__(self, training_config, audio2mel_config, mel2audio_config, unet_config, loss_compute_config):
        self.training_config = training_config
        self.audio2mel_config = audio2mel_config
        self.mel2audio_config = mel2audio_config
        self.unet_config = unet_config
        self.loss_compute_config = loss_compute_config

    def get_configs(self):
        return self.training_config, self.audio2mel_config, self.mel2audio_config, self.unet_config, self.loss_compute_config


def get_experiment_config_debug():
    lr = {'filter_gen': 0.00001, 'filter_disc': 0.00001, 'secret_gen': 0.00001, 'secret_disc': 0.00001}
    loss_compute_config = LossComputeConfig(lamb=100, eps=1e-3, use_entropy_loss=False)
    audio2mel_config = Audio2MelConfig(n_fft=1024, hop_length=256, win_length=1024, sampling_rate=8000, n_mels=80)
    mel2audio_config = Mel2AudioConfig(input_size=audio2mel_config.n_mels, ngf=32, n_residual_layers=3)
    unet_config = UnetConfig(kernel_size=3, embedding_dim=16, noise_dim=10)
    training_config = TrainingConfig(run_name='PCGAN-debug', train_batch_size=8, test_batch_size=8, train_num_workers=1,
                                     test_num_workers=1,
                                     save_interval=1, checkpoint_interval=1, updates_per_evaluation=1,
                                     gradient_accumulation=1, lr=lr, epochs=10, n_samples=10)

    return ExperimentConfig(training_config=training_config, audio2mel_config=audio2mel_config,
                            mel2audio_config=mel2audio_config, unet_config=unet_config,
                            loss_compute_config=loss_compute_config)


def get_experiment_config_fast_run():
    lr = {'filter_gen': 0.00001, 'filter_disc': 0.00001, 'secret_gen': 0.00001, 'secret_disc': 0.00001}
    loss_compute_config = LossComputeConfig(lamb=100, eps=1e-3, use_entropy_loss=False)
    audio2mel_config = Audio2MelConfig(n_fft=1024, hop_length=256, win_length=1024, sampling_rate=8000, n_mels=80)
    mel2audio_config = Mel2AudioConfig(input_size=audio2mel_config.n_mels, ngf=32, n_residual_layers=3)
    unet_config = UnetConfig(kernel_size=3, embedding_dim=16, noise_dim=10)
    training_config = TrainingConfig(run_name='PCGAN-fast_run', train_batch_size=32, test_batch_size=32,
                                     train_num_workers=2, test_num_workers=2,
                                     save_interval=20, checkpoint_interval=20, updates_per_evaluation=20,
                                     gradient_accumulation=1, lr=lr, epochs=10, n_samples=10)

    return ExperimentConfig(training_config=training_config, audio2mel_config=audio2mel_config,
                            mel2audio_config=mel2audio_config, unet_config=unet_config,
                            loss_compute_config=loss_compute_config)


def get_experiment_config_efficient_fast_run():
    lr = {'filter_gen': 0.00001, 'filter_disc': 0.00001, 'secret_gen': 0.00001, 'secret_disc': 0.00001}
    loss_compute_config = LossComputeConfig(lamb=100, eps=1e-3, use_entropy_loss=False)
    audio2mel_config = Audio2MelConfig(n_fft=1024, hop_length=256, win_length=1024, sampling_rate=8000, n_mels=80)
    mel2audio_config = Mel2AudioConfig(input_size=audio2mel_config.n_mels, ngf=32, n_residual_layers=3)
    unet_config = UnetConfig(kernel_size=3, embedding_dim=16, noise_dim=10)
    training_config = TrainingConfig(run_name='PCGAN-efficient_fast_run', train_batch_size=32, test_batch_size=32,
                                     train_num_workers=2, test_num_workers=2,
                                     save_interval=1000000000, checkpoint_interval=100000000, updates_per_evaluation=10,
                                     gradient_accumulation=1, lr=lr, epochs=10, n_samples=10)

    return ExperimentConfig(training_config=training_config, audio2mel_config=audio2mel_config,
                            mel2audio_config=mel2audio_config, unet_config=unet_config,
                            loss_compute_config=loss_compute_config)


def get_experiment_config_pcgan():
    # D_real_loss_weight
    # utility_loss
    # filter_receptive_field == kernel_size?

    lr = {'filter_gen': 0.0001, 'filter_disc': 0.0004, 'secret_gen': 0.0001, 'secret_disc': 0.0004}
    loss_compute_config = LossComputeConfig(lamb=100, eps=1e-3, use_entropy_loss=False)
    audio2mel_config = Audio2MelConfig(n_fft=1024, hop_length=256, win_length=1024, sampling_rate=8000, n_mels=80)
    mel2audio_config = Mel2AudioConfig(input_size=audio2mel_config.n_mels, ngf=32, n_residual_layers=3)
    unet_config = UnetConfig(kernel_size=3, embedding_dim=16, noise_dim=10)
    training_config = TrainingConfig(run_name='PCGAN-pcgan', train_batch_size=64, test_batch_size=64,
                                     train_num_workers=2, test_num_workers=2,
                                     save_interval=1, checkpoint_interval=1, updates_per_evaluation=5,
                                     gradient_accumulation=2, lr=lr, epochs=2, n_samples=10)

    return ExperimentConfig(training_config=training_config, audio2mel_config=audio2mel_config,
                            mel2audio_config=mel2audio_config, unet_config=unet_config,
                            loss_compute_config=loss_compute_config)


def get_experiment_config_low_lr_pcgan():
    # D_real_loss_weight
    # utility_loss
    # filter_receptive_field == kernel_size?

    lr = {'filter_gen': 0.00001, 'filter_disc': 0.00004, 'secret_gen': 0.00001, 'secret_disc': 0.00004}
    loss_compute_config = LossComputeConfig(lamb=100, eps=1e-3, use_entropy_loss=False)
    audio2mel_config = Audio2MelConfig(n_fft=1024, hop_length=256, win_length=1024, sampling_rate=8000, n_mels=80)
    mel2audio_config = Mel2AudioConfig(input_size=audio2mel_config.n_mels, ngf=32, n_residual_layers=3)
    unet_config = UnetConfig(kernel_size=3, embedding_dim=16, noise_dim=10)
    training_config = TrainingConfig(run_name='PCGAN-pcgan', train_batch_size=64, test_batch_size=64,
                                     train_num_workers=2, test_num_workers=2,
                                     save_interval=1, checkpoint_interval=1, updates_per_evaluation=5,
                                     gradient_accumulation=2, lr=lr, epochs=2, n_samples=10)

    return ExperimentConfig(training_config=training_config, audio2mel_config=audio2mel_config,
                            mel2audio_config=mel2audio_config, unet_config=unet_config,
                            loss_compute_config=loss_compute_config)
