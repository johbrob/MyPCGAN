import local_vars
from experiment_runs import ExperimentSetup
from loss_compiling import LossConfig
from neural_networks import AvailableModels, UNet, UNetConfig, AlexNet, AlexNetConfig, ResNet18, ResNetConfig
from neural_networks.audio_mel import AudioMelConfig
from training.initialization import TrainingConfig


class ModelConfig:
    def __init__(self, model, model_config, pretrained_path=None):
        self.model = model
        self.args = model_config
        self.pretrained_path = pretrained_path


Q = [
    ExperimentSetup(
        training_config=TrainingConfig(run_name='debug', epochs=2, train_batch_size=128,
                                       test_batch_size=128, deterministic=True, gradient_accumulation=1,
                                       save_interval=1, checkpoint_interval=1, updates_per_train_log_commit=10,
                                       updates_per_evaluation=50),
        audio_mel_config=AudioMelConfig(),
        filter_gen_config=ModelConfig(UNet, UNetConfig(activation='relu')),
        filter_disc_config=ModelConfig(AlexNet, AlexNetConfig(activation='leaky_relu')),
        secret_gen_config=ModelConfig(UNet, UNetConfig(activation='relu')),
        secret_disc_config=ModelConfig(AlexNet, AlexNetConfig(activation='leaky_relu')),
        label_classifier_config=ModelConfig(ResNet18, ResNetConfig(activation='relu'),
                                            pretrained_path=local_vars.CREMA_D_PRETRAINED_GENDER_CLASSIFIER_PATH),
        secret_classifier_config=ModelConfig(ResNet18, ResNetConfig(activation='relu'),
                                             pretrained_path=local_vars.CREMA_D_PRETRAINED_EMOTION_CLASSIFIER_PATH),
        loss_config=LossConfig())
]

print(len(Q))
