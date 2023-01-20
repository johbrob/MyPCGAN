from experiment_runs import ExperimentSetup
from neural_networks import AvailableModels, UNet, UNetConfig, AlexNet, AlexNetConfig, ResNet18, ResNetConfig
from neural_networks.audio_mel import AudioMelConfig
from training.initialization import TrainingConfig
from architectures.alt_gen.model import OneStepGanConfig, OneStepGAN


class ModelConfig:
    def __init__(self, model, model_config, pretrained_path=None):
        self.model = model
        self.args = model_config
        self.pretrained_path = pretrained_path


Q = [
    ExperimentSetup(
        training_config=TrainingConfig(run_name='debug', epochs=2, train_batch_size=128,
                                       test_batch_size=128, deterministic=True,
                                       save_interval=1, checkpoint_interval=1, updates_per_train_log_commit=10,
                                       updates_per_evaluation=1, do_log=False),
        # audio_mel_config=AudioMelConfig(pretrained_path='neural_networks/pretrained_weights/best_netG_epoch_2120.pt'),
        architecture_config=OneStepGanConfig(
            gen_config=ModelConfig(UNet, UNetConfig(activation='relu')),
            fake_disc_config=ModelConfig(ResNet18, ResNetConfig(activation='leaky_relu')),
            secret_disc_config=ModelConfig(ResNet18, ResNetConfig(activation='leaky_relu')),
            generate_both_secrets=True,
        ))
]
