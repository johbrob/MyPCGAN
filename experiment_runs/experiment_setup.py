# class ExperimentSetup:
#     def __init__(self, training_config, audio_mel_config, unet_config, loss_compute_config):
#         self.training_config = training_config
#         self.audio_mel_config = audio_mel_config
#         self.unet_config = unet_config
#         self.loss_compute_config = loss_compute_config
#
#     def get_configs(self):
#         return self.training_config, self.audio_mel_config, self.unet_config, self.loss_compute_config
#


# class ExperimentSetup:
#     def __init__(self, training_config, audio_mel_config, loss_config, filter_gen_config, filter_disc_config, secret_gen_config,
#                  secret_disc_config, label_classifier_config, secret_classifier_config):
#         self.training = training_config
#         self.audio_mel = audio_mel_config
#         self.loss = loss_config
#
#         self.filter_gen = filter_gen_config
#         self.filter_disc = filter_disc_config
#         self.secret_gen = secret_gen_config
#         self.secret_disc = secret_disc_config
#
#         self.label_classifier = label_classifier_config
#         self.secret_classifier = secret_classifier_config
#
#
#     def get_configs(self):
#         return self.training, self.audio_mel, self.loss, \
#                self.filter_gen, self.filter_gen, self.secret_gen, self.secret_disc, \
#                self.label_classifier, self.secret_classifier
#
#     def __str__(self):
#         return self.training.run_name.rsplit('_')[0]


class ExperimentSetup:
    def __init__(self, training_config, audio_mel_config, architecture_config):
        self.training = training_config
        self.audio_mel = audio_mel_config
        self.architecture = architecture_config


    def get_configs(self):
        return self.training, self.audio_mel, self.architecture

    def __str__(self):
        return self.training.run_name.rsplit('_')[0]
