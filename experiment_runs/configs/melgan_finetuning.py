from datasets import AvailableDatasets
from finetune_melgan.main import FineTuneMelGanConfig, DiscriminatorConfig
from audio_mel_conversion import AudioMelConfig
from neural_networks.whisper_encoder import WhisperSize

Q = [FineTuneMelGanConfig(audio_mel=AudioMelConfig(model_size=WhisperSize.TINY, n_fft=400, hop_length=160, n_mels=80),
                          discriminator=DiscriminatorConfig(numD=3, ndf=16, n_layers=4, downsampling_factor=4,
                                                            lambda_feat=10, cond_disc=True),
                          n_train_samples=128, n_test_samples=64, do_log=False, train_batch_size=4, test_batch_size=1)
     ]
