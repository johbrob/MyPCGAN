from datasets import AvailableDatasets
from finetune_melgan.main import FineTuneMelGanConfig, DiscriminatorConfig, Converter
from audio_mel_conversion import AudioMelConfig
from neural_networks.whisper_encoder import WhisperSize

Q = [FineTuneMelGanConfig(audio_mel=AudioMelConfig(model_size=WhisperSize.TINY, n_fft=400, hop_length=160, n_mels=80),
                          discriminator=DiscriminatorConfig(numD=3, ndf=16, n_layers=4, downsampling_factor=4,
                                                            lambda_feat=10, cond_disc=True),
                          do_log=True, train_batch_size=32, test_batch_size=1, converter=Converter.CUT_FIRST,
                          pretrained_path='neural_networks/pretrained_weights/multi_speaker.pt',
                          ),
     FineTuneMelGanConfig(audio_mel=AudioMelConfig(model_size=WhisperSize.TINY, n_fft=400, hop_length=160, n_mels=80),
                          discriminator=DiscriminatorConfig(numD=3, ndf=16, n_layers=4, downsampling_factor=4,
                                                            lambda_feat=10, cond_disc=True),
                          do_log=True, train_batch_size=32, test_batch_size=1, converter=Converter.CUT_LAST,
                          pretrained_path='neural_networks/pretrained_weights/multi_speaker.pt',
                          ),
     ]
