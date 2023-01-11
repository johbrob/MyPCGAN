import local_vars
from experiment_runs import ExperimentSetup
from neural_networks import AvailableModels, UNet, UNetConfig, AlexNet, AlexNetConfig, ResNet18, ResNetConfig
from neural_networks.audio_mel import AudioMelConfig
from training.initialization import TrainingConfig
from architectures import PCGANConfig, PCGAN
from architectures.pcgan.loss_computations import LossConfig
from datasets import AvailableDatasets


class ModelConfig:
    def __init__(self, model, model_config, pretrained_path=None):
        self.model = model
        self.args = model_config
        self.pretrained_path = pretrained_path


Q = [
    ExperimentSetup(
        training_config=TrainingConfig(
            run_name='BASE_entropy', dataset=AvailableDatasets.CremaD, epochs=1000, train_batch_size=128,
            test_batch_size=128, deterministic=False, gradient_accumulation=1, save_interval=10, checkpoint_interval=10,
            updates_per_train_log_commit=10, updates_per_evaluation=50, do_log=True, test_num_workers=0),
        # audio_mel_config=AudioMelConfig(
        #     pretrained_path='neural_networks/pretrained_weights/best_netG_epoch_2120.pt'),
        architecture_config=PCGANConfig(
            filter_gen_config=ModelConfig(UNet, UNetConfig(activation='relu')),
            filter_disc_config=ModelConfig(ResNet18, ResNetConfig(activation='relu')),
            secret_gen_config=ModelConfig(UNet, UNetConfig(activation='relu')),
            secret_disc_config=ModelConfig(ResNet18, ResNetConfig(activation='relu')),
            label_classifier_config=ModelConfig(ResNet18, ResNetConfig(activation='relu'),
                                                pretrained_path=local_vars.CREMA_D_PRETRAINED_GENDER_CLASSIFIER_PATH),
            secret_classifier_config=ModelConfig(ResNet18, ResNetConfig(activation='relu'),
                                                 pretrained_path=local_vars.CREMA_D_PRETRAINED_EMOTION_CLASSIFIER_PATH),
            loss_config=LossConfig(filter_entropy_loss=True)
        )
    ),
]
