from loss_compiling import LossConfig
from nn.models import UnetConfig
from nn.audio_mel import AudioMelConfig
from training.initialization import TrainingConfig


class ExperimentConfig:
    def __init__(self, training_config, audio_mel_config, unet_config, loss_compute_config):
        self.training_config = training_config
        self.audio_mel_config = audio_mel_config
        self.unet_config = unet_config
        self.loss_compute_config = loss_compute_config

    def get_configs(self):
        return self.training_config, self.audio_mel_config, self.unet_config, self.loss_compute_config


lr = {'filter_gen': 0.0001, 'filter_disc': 0.0004, 'secret_gen': 0.0001, 'secret_disc': 0.0004}


def create_debug_config():
    return ExperimentConfig(
        training_config=TrainingConfig(lr=lr, run_name='debug', train_batch_size=8, test_batch_size=8,
                                       train_num_workers=1, test_num_workers=1, updates_per_evaluation=1, do_log=False),
        audio_mel_config=AudioMelConfig(), unet_config=UnetConfig(), loss_compute_config=LossConfig())


def create_github_default_config():
    # Missing settings: D_real_loss_weight, utility_loss, filter_receptive_field == kernel_size?
    return ExperimentConfig(training_config=TrainingConfig(lr={k: 10 * v for k, v in lr.items()},
                                                           run_name='github_default', epochs=10),
                            audio_mel_config=AudioMelConfig(), unet_config=UnetConfig(),
                            loss_compute_config=LossConfig())


# def create_github_lower_lr_config():
#     # Missing settings: D_real_loss_weight, utility_loss, filter_receptive_field == kernel_size?
#     return ExperimentConfig(training_config=TrainingConfig(lr=lr, run_name='github_lower_lr', epochs=50),
#                             audio_mel_config=AudioMelConfig(), unet_config=UnetConfig(),
#                             loss_compute_config=LossConfig())

def create_github_lower_lr_config():
    # Missing settings: D_real_loss_weight, utility_loss, filter_receptive_field == kernel_size?
    return ExperimentConfig(training_config=TrainingConfig(lr=lr, run_name='github_lower_lr', epochs=50),
                            audio_mel_config=AudioMelConfig(), unet_config=UnetConfig(),
                            loss_compute_config=LossConfig(gamma=100, epsilon=1e-3))
