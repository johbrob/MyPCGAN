from architectures.utils import preprocess_spectrograms, create_model_from_config
from architectures.pcgan.loss_computations import compute_losses, LossConfig
from architectures.pcgan.metrics_computations import compute_metrics
from utils import Mode
import torch
import glob
import os


class HLoss(torch.nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = torch.nn.functional.softmax(x, dim=1) * torch.nn.functional.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b


class PCGANConfig:
    def __init__(self, filter_gen_config, filter_disc_config, secret_gen_config, secret_disc_config,
                 label_classifier_config, secret_classifier_config, lr=None, betas=None, generate_both_secrets=True,
                 filter_gen_label_smoothing=0, secret_gen_label_smoothing=0,
                 filter_disc_label_smoothing=0, secret_disc_label_smoothing=0, loss_config=LossConfig()):

        self.architecture = PCGAN

        self.filter_gen = filter_gen_config
        self.filter_disc = filter_disc_config
        self.secret_gen = secret_gen_config
        self.secret_disc = secret_disc_config

        self.label_classifier = label_classifier_config
        self.secret_classifier = secret_classifier_config

        if lr is None:
            self.lr = {'filter_gen': 0.0001, 'filter_disc': 0.0004, 'secret_gen': 0.0001, 'secret_disc': 0.0004}
        else:
            self.lr = lr

        if betas is None:
            self.betas = {'filter_gen': (0.5, 0.999), 'filter_disc': (0.5, 0.999),
                          'secret_gen': (0.5, 0.999), 'secret_disc': (0.5, 0.999)}
        else:
            self.betas = betas

        self.generate_both_secrets = generate_both_secrets
        self.filter_gen_label_smoothing = filter_gen_label_smoothing
        self.secret_gen_label_smoothing = secret_gen_label_smoothing
        self.filter_disc_label_smoothing = filter_disc_label_smoothing
        self.secret_disc_label_smoothing = secret_disc_label_smoothing

        self.loss = loss_config


class PCGAN:
    def __init__(self, config, device):

        config.filter_gen.args.image_width = config.image_width
        config.filter_gen.args.image_height = config.image_height
        config.filter_gen.args.n_classes = config.n_genders
        config.filter_disc.args.n_classes = config.n_genders
        config.secret_gen.args.image_width = config.image_width
        config.secret_gen.args.image_height = config.image_height
        config.secret_gen.args.n_classes = config.n_genders
        config.secret_disc.args.n_classes = config.n_genders + 1
        config.label_classifier.args.n_classes = config.n_genders
        config.secret_classifier.args.n_classes = config.n_labels

        self.loss_funcs = {'filter_gen_distortion': torch.nn.L1Loss(), 'filter_gen_entropy': HLoss(),
                           'filter_gen_adversarial': torch.nn.CrossEntropyLoss(
                               label_smoothing=config.filter_gen_label_smoothing),
                           'filter_disc': torch.nn.CrossEntropyLoss(
                               label_smoothing=config.filter_disc_label_smoothing),
                           'secret_gen_distortion': torch.nn.L1Loss(), 'secret_gen_entropy': HLoss(),
                           'secret_gen_adversarial': torch.nn.CrossEntropyLoss(
                               label_smoothing=config.secret_gen_label_smoothing),
                           'secret_disc': torch.nn.CrossEntropyLoss(
                               label_smoothing=config.secret_disc_label_smoothing)}

        self.filter_gen = create_model_from_config(config.filter_gen).to(device)
        self.filter_disc = create_model_from_config(config.filter_disc).to(device)
        self.secret_gen = create_model_from_config(config.secret_gen).to(device)
        self.secret_disc = create_model_from_config(config.secret_disc).to(device)
        self.label_classifier = create_model_from_config(config.label_classifier).to(device)
        self.secret_classifier = create_model_from_config(config.secret_classifier).to(device)

        if config.label_classifier.pretrained_path:
            self.label_classifier.model.load_state_dict(
                torch.load(config.label_classifier.pretrained_path, map_location=torch.device('cpu')))
        if config.secret_classifier.pretrained_path:
            self.secret_classifier.model.load_state_dict(
                torch.load(config.secret_classifier.pretrained_path, map_location=torch.device('cpu')))

        self.lr = config.lr
        betas = config.betas
        self.optimizers = {
            'filter_gen': torch.optim.Adam(self.filter_gen.parameters(), self.lr['filter_gen'], betas['filter_gen']),
            'filter_disc': torch.optim.Adam(self.filter_disc.parameters(), self.lr['filter_disc'],
                                            betas['filter_disc']),
            'secret_gen': torch.optim.Adam(self.secret_gen.parameters(), self.lr['secret_gen'], betas['secret_gen']),
            'secret_disc': torch.optim.Adam(self.secret_disc.parameters(), self.lr['secret_disc'], betas['secret_disc'])
        }

        self.loss_config = config.loss
        self.generate_both_secrets = config.generate_both_secrets

        self.device = device

    def _filter_gen_forward_pass(self, mels, secrets):
        filter_z = torch.randn(mels.shape[0], self.filter_gen.noise_dim).to(mels.device)
        filtered_mels = self.filter_gen(mels, filter_z, secrets.long())  # (bsz, 1, n_mels, frames)
        filtered_secret_preds_gen = self.filter_disc(filtered_mels, frozen=True)  # (bsz, n_secret)
        return {'filtered_mel': filtered_mels, 'filtered_secret_score': filtered_secret_preds_gen}

    def _secret_gen_forward_pass(self, mels, filtered_mel):
        secret_z = torch.randn(mels.shape[0], self.secret_gen.noise_dim).to(mels.device)
        fake_secret_gen = torch.randint(0, 1, mels.shape[0:1]).to(mels.device)  # (bsz,)
        fake_mel = self.secret_gen(filtered_mel.detach(), secret_z, fake_secret_gen)  # (bsz, 1, n_mels, frames)
        fake_secret_preds_gen = self.secret_disc(fake_mel, frozen=True)  # (bsz, n_secrets + 1)
        secret_gen_output = {'fake_secret': fake_secret_gen, 'faked_mel': fake_mel,
                             'fake_secret_score': fake_secret_preds_gen}

        if self.generate_both_secrets:
            # (bsz, 1, n_mels, frames)
            alt_fake_mel = self.secret_gen(filtered_mel.detach(), secret_z, 1 - fake_secret_gen, frozen=True)
            # (bsz, n_secrets + 1)
            alt_fake_secret_preds_gen = self.secret_disc(alt_fake_mel, frozen=True)
            secret_gen_output.update(
                {'alt_faked_mel': alt_fake_mel, 'alt_fake_secret_score': alt_fake_secret_preds_gen})

        return secret_gen_output

    def _filter_disc_forward_pass(self, mels, filtered_mels):
        filtered_secret_preds_disc = self.filter_disc(filtered_mels.detach())  # (bsz, n_secret)
        unfiltered_secret_preds_disc = self.filter_disc(mels.detach(), frozen=True)  # (bsz, n_secret)
        return {'filtered_secret_score': filtered_secret_preds_disc,
                'unfiltered_secret_score': unfiltered_secret_preds_disc}

    def _secret_disc_forward_pass(self, mels, fake_mel):
        fake_secret_preds_disc = self.secret_disc(fake_mel.detach().clone())  # (bsz, n_secrets + 1)
        real_secret_preds_disc = self.secret_disc(mels)  # (bsz, n_secrets + 1)
        fake_secret_disc = 2 * torch.ones(mels.size(0), requires_grad=False, dtype=torch.int64).to(
            mels.device)  # (bsz,)
        return {'fake_secret_score': fake_secret_preds_disc, 'real_secret_score': real_secret_preds_disc,
                'fake_secret': fake_secret_disc}

    def forward_pass(self, audio, secrets, labels):

        mels = self.audio_mel_converter.audio2mel(audio).detach()  # mels: (bsz, n_mels, frames)
        mels, means, stds = preprocess_spectrograms(mels)
        mels = mels.unsqueeze(dim=1).to(self.device)  # mels: (bsz, 1, n_mels, frames)
        assert mels.dim() == 4

        filter_gen_output = self._filter_gen_forward_pass(mels, secrets)
        secret_gen_output = self._secret_gen_forward_pass(mels, filter_gen_output['filtered_mel'])
        filter_disc_output = self._filter_disc_forward_pass(mels, filter_gen_output['filtered_mel'])
        secret_disc_output = self._secret_disc_forward_pass(mels, secret_gen_output['faked_mel'])

        label_preds = self.label_classifier(secret_gen_output['faked_mel'])
        secret_preds = self.secret_classifier(secret_gen_output['faked_mel'])
        secret_disc_output.update({'label_score': label_preds, 'secret_score': secret_preds})

        losses = compute_losses(self.loss_funcs, mels, secrets, filter_gen_output, filter_disc_output,
                                secret_gen_output, secret_disc_output, self.loss_config)
        batch_metrics = compute_metrics(mels, secrets, labels, filter_gen_output, filter_disc_output, secret_gen_output,
                                        secret_disc_output, losses, self.loss_funcs)

        return batch_metrics, losses

    def generate_sample(self, audio, secret, label, id, audio_mel_converter, epoch):

        mel = self.audio_mel_converter.audio2mel(audio).detach()  # mels: (bsz, n_mels, frames)
        mel, mean, std = preprocess_spectrograms(mel)
        mel = mel.unsqueeze(dim=1).to(self.device)  # mels: (bsz, 1, n_mels, frames)
        assert mel.dim() == 4

        # filter_gen
        filter_z = torch.randn(mel.shape[0], self.filter_gen.noise_dim).to(self.device)
        filtered = self.filter_gen(mel, filter_z, secret.long())

        # secret_gen
        secret_z = torch.randn(mel.shape[0], self.secret_gen.noise_dim).to(self.device)
        fake_secret_male = torch.ones(mel.shape[0], requires_grad=False, dtype=torch.int64).to(self.device)
        fake_secret_female = torch.zeros(mel.shape[0], requires_grad=False, dtype=torch.int64).to(self.device)
        fake_mel_male = self.secret_gen(filtered, secret_z, fake_secret_male)
        fake_mel_female = self.secret_gen(filtered, secret_z, fake_secret_female)

        # predict label
        pred_label_male = torch.argmax(self.label_classifier(fake_mel_male).data, 1)
        pred_label_female = torch.argmax(self.label_classifier(fake_mel_female).data, 1)

        # predict secret
        pred_secret_male = torch.argmax(self.secret_classifier(fake_mel_male).data, 1)
        pred_secret_female = torch.argmax(self.secret_classifier(fake_mel_female).data, 1)

        unnormalized_filtered_mel = (torch.squeeze(filtered.cpu(), 1) * 3 * std + mean)
        unnormalized_fake_mel_male = (torch.squeeze(fake_mel_male.cpu(), 1) * 3 * std + mean)
        unnormalized_fake_mel_female = (torch.squeeze(fake_mel_female.cpu(), 1) * 3 * std + mean)
        unnormalized_mel = torch.squeeze(mel.cpu() * 3 * std + mean)

        print(unnormalized_mel.shape, unnormalized_mel.squeeze().shape)

        a2m2a_audio = audio_mel_converter.mel2audio(unnormalized_mel).squeeze()
        filtered_audio = audio_mel_converter.mel2audio(unnormalized_filtered_mel.squeeze()).squeeze()
        audio_male = audio_mel_converter.mel2audio(unnormalized_fake_mel_male.squeeze()).squeeze()
        audio_female = audio_mel_converter.mel2audio(unnormalized_fake_mel_female.squeeze()).squeeze()

        print(audio.shape, a2m2a_audio.shape, filtered_audio.shape, audio_male.shape, audio_female.shape)
        speaker_digit_str = f'speaker_{id.item()}_label_{label.item()}'
        speaker_digit_epoch_str = speaker_digit_str + f'_epoch_{epoch}'

        def build_str(gnd, lbl): return speaker_digit_epoch_str + f'_sampled_gender_{gnd}_predicted_label_{lbl}.wav'

        filtered_audio_file = speaker_digit_epoch_str + '_filtered.wav'
        a2m2a_file = speaker_digit_epoch_str + '_a2m2a.wav'
        male_audio_file = build_str('male', pred_label_male.item())
        female_audio_file = build_str('female', pred_label_female.item())
        original_audio_file = speaker_digit_str + '.wav'

        return {original_audio_file: audio.squeeze().cpu(),
                filtered_audio_file: filtered_audio,
                a2m2a_file: a2m2a_audio,
                male_audio_file: audio_male,
                female_audio_file: audio_female}

        # return {'mel': mel, 'filter_z': filter_z, 'filtered_spectrogram': filtered, 'secret_z': secret_z,
        #         'fake_secret_male': fake_secret_male, 'fake_secret_female': fake_secret_female,
        #         'fake_mel_male': fake_mel_male, 'fake_mel_female': fake_mel_female,
        #         'pred_label_male': pred_label_male, 'pred_label_female': pred_label_female,
        #         'pred_secret_male': pred_secret_male, 'pred_secret_female': pred_secret_female}

    def set_mode(self, mode):
        if mode == Mode.TRAIN:
            self.filter_gen.train()
            self.filter_disc.train()
            self.secret_gen.train()
            self.secret_disc.train()
        elif mode == Mode.EVAL:
            self.filter_gen.eval()
            self.filter_disc.eval()
            self.secret_gen.eval()
            self.secret_disc.eval()

    def save(self, checkpoint_dir, epoch):
        models = {'filter_gen': self.filter_gen, 'filter_disc': self.filter_disc,
                  'secret_gen': self.secret_gen, 'secret_disc': self.secret_disc}

        old_checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, '*latest*')))
        if old_checkpoints:
            for i, _ in enumerate(old_checkpoints):
                os.remove(old_checkpoints[i])
        for k, v in models.items():
            if k == 'label_classifier' or k == 'secret_classifier':
                continue
            torch.save(v.state_dict(), os.path.join(checkpoint_dir, f'{k}_epoch_{epoch}.pt'))
            torch.save(v.state_dict(), os.path.join(checkpoint_dir, f'{k}_latest_epoch_{epoch}.pt'))
        for k, v in self.optimizers.items():
            torch.save(v.state_dict(), os.path.join(checkpoint_dir, f'optimizer_{k}_epoch_{epoch}.pt'))
            torch.save(v.state_dict(), os.path.join(checkpoint_dir, f'optimizer_{k}_latest_epoch_{epoch}.pt'))