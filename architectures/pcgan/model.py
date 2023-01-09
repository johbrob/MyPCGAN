import torch

class PCGAN:
    def __init__(self):
        label_classifier = create_model_from_config(experiment_setup.label_classifier).to(device)
        if experiment_setup.label_classifier.pretrained_path:
            label_classifier.model.load_state_dict(
                torch.load(experiment_setup.label_classifier.pretrained_path, map_location=torch.device('cpu')))

        secret_classifier = create_model_from_config(experiment_setup.secret_classifier).to(device)
        if experiment_setup.secret_classifier.pretrained_path:
            secret_classifier.model.load_state_dict(
                torch.load(experiment_setup.secret_classifier.pretrained_path, map_location=torch.device('cpu')))

        loss_funcs = {'filter_gen_distortion': torch.nn.L1Loss(), 'filter_gen_entropy': HLoss(),
                      'filter_gen_adversarial': torch.nn.CrossEntropyLoss(
                          label_smoothing=experiment_setup.loss.filter_gen_label_smoothing),
                      'filter_disc': torch.nn.CrossEntropyLoss(
                          label_smoothing=experiment_setup.loss.filter_disc_label_smoothing),
                      'secret_gen_distortion': torch.nn.L1Loss(), 'secret_gen_entropy': HLoss(),
                      'secret_gen_adversarial': torch.nn.CrossEntropyLoss(
                          label_smoothing=experiment_setup.loss.secret_gen_label_smoothing),
                      'secret_disc': torch.nn.CrossEntropyLoss(
                          label_smoothing=experiment_setup.loss.secret_disc_label_smoothing)}

        models = {
            'filter_gen': create_model_from_config(experiment_setup.filter_gen).to(device),
            'filter_disc': create_model_from_config(experiment_setup.filter_disc).to(device),
            'secret_gen': create_model_from_config(experiment_setup.secret_gen).to(device),
            'secret_disc': create_model_from_config(experiment_setup.secret_disc).to(device),
            'label_classifier': label_classifier,
            'secret_classifier': secret_classifier
        }

        lr = experiment_setup.training.lr
        betas = experiment_setup.training.betas
        optimizers = {
            'filter_gen': torch.optim.Adam(models['filter_gen'].parameters(), lr['filter_gen'], betas['filter_gen']),
            'filter_disc': torch.optim.Adam(models['filter_disc'].parameters(), lr['filter_disc'],
                                            betas['filter_disc']),
            'secret_gen': torch.optim.Adam(models['secret_gen'].parameters(), lr['secret_gen'], betas['secret_gen']),
            'secret_disc': torch.optim.Adam(models['secret_disc'].parameters(), lr['secret_disc'], betas['secret_disc'])
        }
        return loss_funcs, models, optimizers

    def filter_gen_forward_pass(self, filter_gen, filter_disc, mels, secrets):
        filter_z = torch.randn(mels.shape[0], filter_gen.noise_dim).to(mels.device)
        filtered_mels = filter_gen(mels, filter_z, secrets.long())  # (bsz, 1, n_mels, frames)
        filtered_secret_preds_gen = filter_disc(filtered_mels, frozen=True)  # (bsz, n_secret)
        return {'filtered_mel': filtered_mels, 'filtered_secret_score': filtered_secret_preds_gen}


    def secret_gen_forward_pass(self, secret_gen, secret_disc, mels, filtered_mel):
        secret_z = torch.randn(mels.shape[0], secret_gen.noise_dim).to(mels.device)
        fake_secret_gen = torch.randint(0, 1, mels.shape[0:1]).to(mels.device)  # (bsz,)
        fake_mel = secret_gen(filtered_mel.detach(), secret_z, fake_secret_gen)  # (bsz, 1, n_mels, frames)
        fake_secret_preds_gen = secret_disc(fake_mel, frozen=True)  # (bsz, n_secrets + 1)
        secret_gen_output = {'fake_secret': fake_secret_gen, 'faked_mel': fake_mel,
                             'fake_secret_score': fake_secret_preds_gen}

        generate_both_genders = True
        if generate_both_genders:
            alt_fake_mel = secret_gen(filtered_mel.detach(), secret_z, 1 - fake_secret_gen,
                                      frozen=True)  # (bsz, 1, n_mels, frames)
            alt_fake_secret_preds_gen = secret_disc(alt_fake_mel, frozen=True)  # (bsz, n_secrets + 1)
            secret_gen_output.update({'alt_faked_mel': alt_fake_mel, 'alt_fake_secret_score': alt_fake_secret_preds_gen})

        return secret_gen_output


    def filter_disc_forward_pass(self, filter_disc, mels, filtered_mels):
        filtered_secret_preds_disc = filter_disc(filtered_mels.detach())  # (bsz, n_secret)
        unfiltered_secret_preds_disc = filter_disc(mels.detach(), frozen=True)  # (bsz, n_secret)
        return {'filtered_secret_score': filtered_secret_preds_disc,
                'unfiltered_secret_score': unfiltered_secret_preds_disc}


    def secret_disc_forward_pass(self, secret_disc, mels, fake_mel):
        fake_secret_preds_disc = secret_disc(fake_mel.detach().clone())  # (bsz, n_secrets + 1)
        real_secret_preds_disc = secret_disc(mels)  # (bsz, n_secrets + 1)
        fake_secret_disc = 2 * torch.ones(mels.size(0), requires_grad=False, dtype=torch.int64).to(mels.device)  # (bsz,)
        return {'fake_secret_score': fake_secret_preds_disc, 'real_secret_score': real_secret_preds_disc,
                'fake_secret': fake_secret_disc}


    def forward_pass(self, models, mels, secrets):
        assert mels.dim() == 4
        filter_gen_output = self.filter_gen_forward_pass(models['filter_gen'], models['filter_disc'], mels, secrets)
        secret_gen_output = self.secret_gen_forward_pass(models['secret_gen'], models['secret_disc'], mels,
                                                    filter_gen_output['filtered_mel'])
        filter_disc_output = self.filter_disc_forward_pass(models['filter_disc'], mels, filter_gen_output['filtered_mel'])
        secret_disc_output = self.secret_disc_forward_pass(models['secret_disc'], mels, secret_gen_output['faked_mel'])

        label_preds = models['label_classifier'](secret_gen_output['faked_mel'])
        secret_preds = models['secret_classifier'](secret_gen_output['faked_mel'])
        secret_disc_output.update({'label_score': label_preds, 'secret_score': secret_preds})

        return filter_gen_output, filter_disc_output, secret_gen_output, secret_disc_output