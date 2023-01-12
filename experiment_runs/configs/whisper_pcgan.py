import local_vars
from experiment_runs import ExperimentSetup
from neural_networks import AvailableModels, UNet, UNetConfig, AlexNet, AlexNetConfig, ResNet18, ResNetConfig, \
    WhisperEncoderConfig, WhisperSize
from audio_mel_conversion import AudioMelConfig, LibRosaAudio2Mel, LibRosaMel2Audio, WhisperAudio2Mel
from training.initialization import TrainingConfig
from architectures import WhisperPcgan, WhistperPcganConfig
from architectures.whisper_pcgan.loss_computations import LossConfig
from datasets import AvailableDatasets


class ModelConfig:
    def __init__(self, model, model_config, pretrained_path=None):
        self.model = model
        self.args = model_config
        self.pretrained_path = pretrained_path


Q = [
    ExperimentSetup(
        training_config=TrainingConfig(
            run_name='Whisper-PCGAN', dataset=AvailableDatasets.CremaD, epochs=1000, train_batch_size=4,
            test_batch_size=4, deterministic=False, gradient_accumulation=4, save_interval=1, checkpoint_interval=10,
            updates_per_train_log_commit=10, updates_per_evaluation=50, do_log=True, test_num_workers=0,
            n_train_samples=128, n_test_samples=64),
        architecture_config=WhistperPcganConfig(
            filter_gen_config=ModelConfig(UNet, UNetConfig(activation='relu')),
            filter_disc_config=ModelConfig(ResNet18, ResNetConfig(activation='relu')),
            secret_gen_config=ModelConfig(UNet, UNetConfig(activation='relu')),
            secret_disc_config=ModelConfig(ResNet18, ResNetConfig(activation='relu')),
            whisper_config=WhisperEncoderConfig(WhisperSize.TINY, 16000),
            audio2mel_config=ModelConfig(WhisperAudio2Mel, AudioMelConfig()),
            mel2audio_config=ModelConfig(LibRosaMel2Audio, AudioMelConfig()),
            loss_config=LossConfig(filter_entropy_loss=True)
        )
    ),
]
