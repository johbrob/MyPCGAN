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
        training_config=TrainingConfig(run_name='BASE_All_ReLU', epochs=1000, train_batch_size=128,
                                       test_batch_size=128, deterministic=True, gradient_accumulation=1,
                                       save_interval=10, checkpoint_interval=10, updates_per_train_log_commit=10,
                                       updates_per_evaluation=50, do_log=True),
        audio_mel_config=AudioMelConfig(pretrained_path='neural_networks/pretrained_weights/multi_speaker.pt'),
        filter_gen_config=ModelConfig(UNet, UNetConfig(activation='relu')),
        filter_disc_config=ModelConfig(ResNet18, ResNetConfig(activation='relu')),
        secret_gen_config=ModelConfig(UNet, UNetConfig(activation='relu')),
        secret_disc_config=ModelConfig(ResNet18, ResNetConfig(activation='relu')),
        label_classifier_config=ModelConfig(ResNet18, ResNetConfig(activation='relu'),
                                            pretrained_path=local_vars.CREMA_D_PRETRAINED_GENDER_CLASSIFIER_PATH),
        secret_classifier_config=ModelConfig(ResNet18, ResNetConfig(activation='relu'),
                                             pretrained_path=local_vars.CREMA_D_PRETRAINED_EMOTION_CLASSIFIER_PATH),
        loss_config=LossConfig()
    ),
    ExperimentSetup(
        training_config=TrainingConfig(run_name='BASE_Disc_LeakyReLU', epochs=1000, train_batch_size=128,
                                       test_batch_size=128, deterministic=True, gradient_accumulation=1,
                                       save_interval=10, checkpoint_interval=10, updates_per_train_log_commit=10,
                                       updates_per_evaluation=50, do_log=True),
        audio_mel_config=AudioMelConfig(pretrained_path='neural_networks/pretrained_weights/multi_speaker.pt'),
        filter_gen_config=ModelConfig(UNet, UNetConfig(activation='relu')),
        filter_disc_config=ModelConfig(ResNet18, ResNetConfig(activation='leaky_relu')),
        secret_gen_config=ModelConfig(UNet, UNetConfig(activation='relu')),
        secret_disc_config=ModelConfig(ResNet18, ResNetConfig(activation='leaky_relu')),
        label_classifier_config=ModelConfig(ResNet18, ResNetConfig(activation='relu'),
                                            pretrained_path=local_vars.CREMA_D_PRETRAINED_GENDER_CLASSIFIER_PATH),
        secret_classifier_config=ModelConfig(ResNet18, ResNetConfig(activation='relu'),
                                             pretrained_path=local_vars.CREMA_D_PRETRAINED_EMOTION_CLASSIFIER_PATH),
        loss_config=LossConfig()
    ),
    ExperimentSetup(
        training_config=TrainingConfig(run_name='BASE_All_LeakyReLU', epochs=1000, train_batch_size=128,
                                       test_batch_size=128, deterministic=True, gradient_accumulation=1,
                                       save_interval=10, checkpoint_interval=10, updates_per_train_log_commit=10,
                                       updates_per_evaluation=50, do_log=True),
        audio_mel_config=AudioMelConfig(pretrained_path='neural_networks/pretrained_weights/multi_speaker.pt'),
        filter_gen_config=ModelConfig(UNet, UNetConfig(activation='leaky_relu')),
        filter_disc_config=ModelConfig(ResNet18, ResNetConfig(activation='leaky_relu')),
        secret_gen_config=ModelConfig(UNet, UNetConfig(activation='leaky_relu')),
        secret_disc_config=ModelConfig(ResNet18, ResNetConfig(activation='leaky_relu')),
        label_classifier_config=ModelConfig(ResNet18, ResNetConfig(activation='relu'),
                                            pretrained_path=local_vars.CREMA_D_PRETRAINED_GENDER_CLASSIFIER_PATH),
        secret_classifier_config=ModelConfig(ResNet18, ResNetConfig(activation='relu'),
                                             pretrained_path=local_vars.CREMA_D_PRETRAINED_EMOTION_CLASSIFIER_PATH),
        loss_config=LossConfig()
    ),
    # ExperimentSetup(
    #     training_config=TrainingConfig(run_name='BASE_All_LeakyReLU_eps_1e-4', epochs=1000, train_batch_size=128,
    #                                    test_batch_size=128, deterministic=True, gradient_accumulation=1,
    #                                    save_interval=10, checkpoint_interval=10, updates_per_train_log_commit=10,
    #                                    updates_per_evaluation=50, do_log=True),
    #     audio_mel_config=AudioMelConfig(pretrained_path='neural_networks/pretrained_weights/multi_speaker.pt'),
    #     filter_gen_config=ModelConfig(UNet, UNetConfig(activation='leaky_relu')),
    #     filter_disc_config=ModelConfig(ResNet18, ResNetConfig(activation='leaky_relu')),
    #     secret_gen_config=ModelConfig(UNet, UNetConfig(activation='leaky_relu')),
    #     secret_disc_config=ModelConfig(ResNet18, ResNetConfig(activation='leaky_relu')),
    #     label_classifier_config=ModelConfig(ResNet18, ResNetConfig(activation='relu'),
    #                                         pretrained_path=local_vars.CREMA_D_PRETRAINED_GENDER_CLASSIFIER_PATH),
    #     secret_classifier_config=ModelConfig(ResNet18, ResNetConfig(activation='relu'),
    #                                          pretrained_path=local_vars.CREMA_D_PRETRAINED_EMOTION_CLASSIFIER_PATH),
    #     loss_config=LossConfig(filter_epsilon=1e-4, secret_epsilon=1e-4)
    # ),
    # ExperimentSetup(
    #     training_config=TrainingConfig(run_name='BASE_All_LeakyReLU_eps_1e-2', epochs=1000, train_batch_size=128,
    #                                    test_batch_size=128, deterministic=True, gradient_accumulation=1,
    #                                    save_interval=10, checkpoint_interval=10, updates_per_train_log_commit=10,
    #                                    updates_per_evaluation=50, do_log=True),
    #     audio_mel_config=AudioMelConfig(pretrained_path='neural_networks/pretrained_weights/multi_speaker.pt'),
    #     filter_gen_config=ModelConfig(UNet, UNetConfig(activation='leaky_relu')),
    #     filter_disc_config=ModelConfig(ResNet18, ResNetConfig(activation='leaky_relu')),
    #     secret_gen_config=ModelConfig(UNet, UNetConfig(activation='leaky_relu')),
    #     secret_disc_config=ModelConfig(ResNet18, ResNetConfig(activation='leaky_relu')),
    #     label_classifier_config=ModelConfig(ResNet18, ResNetConfig(activation='relu'),
    #                                         pretrained_path=local_vars.CREMA_D_PRETRAINED_GENDER_CLASSIFIER_PATH),
    #     secret_classifier_config=ModelConfig(ResNet18, ResNetConfig(activation='relu'),
    #                                          pretrained_path=local_vars.CREMA_D_PRETRAINED_EMOTION_CLASSIFIER_PATH),
    #     loss_config=LossConfig(filter_epsilon=1e-4, secret_epsilon=1e-4)
    # ),
]

