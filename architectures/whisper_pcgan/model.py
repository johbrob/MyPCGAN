from architectures.utils import preprocess_spectrograms, create_model_from_config
from architectures.whisper_pcgan.loss_computations import compute_losses, LossConfig
from architectures.whisper_pcgan.metrics_computations import compute_metrics
from neural_networks.whisper_encoder import WhisperEncoder
from utils import Mode
import torch
import glob
import os
import librosa
import GPUtil

class HLoss(torch.nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = torch.nn.functional.softmax(x, dim=1) * torch.nn.functional.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b


class WhistperPcganConfig:
    def __init__(self, filter_gen_config, filter_disc_config, secret_gen_config, secret_disc_config,
                 # label_classifier_config, secret_classifier_config,
                 whisper_config, audio2mel_config, mel2audio_config,
                 lr=None, betas=None, generate_both_secrets=True, filter_gen_label_smoothing=0,
                 secret_gen_label_smoothing=0, filter_disc_label_smoothing=0, secret_disc_label_smoothing=0,
                 loss_config=LossConfig()):

        self.architecture = WhisperPcgan

        self.filter_gen = filter_gen_config
        self.filter_disc = filter_disc_config
        self.secret_gen = secret_gen_config
        self.secret_disc = secret_disc_config

        # self.label_classifier = label_classifier_config
        # self.secret_classifier = secret_classifier_config

        self.whisper = whisper_config

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

        self.audio2mel = audio2mel_config
        self.mel2audio = mel2audio_config


class WhisperPcgan:
    def __init__(self, config, device):

        self.audio2mel = config.audio2mel.model(config.audio2mel.args)
        self.mel2audio = config.mel2audio.model(config.mel2audio.args)
        if config.mel2audio.pretrained_path:
            self.mel2audio.load_state_dict(
                torch.load(config.mel2audio.pretrained_path, map_location=torch.device('cpu')))

        self.audio_encoder = config.whisper.model(config.whisper.args, device)

        # self.audio_encoder = WhisperEncoder(config.whisper, device)
        # self.audio2mel = create_model_from_config(config.audio2mel).to(device)
        # self.mel2audio = create_model_from_config(config.mel2audio).to(device)

        image_width, image_height = self.audio2mel.output_shape(config.data_info['sample_data'][0])
        self.n_secrets = config.data_info['n_secrets']
        self.n_labels = config.data_info['n_labels']

        config.filter_gen.args.image_width = image_width
        config.filter_gen.args.image_height = image_height
        config.filter_gen.args.n_classes = self.n_secrets
        config.filter_disc.args.n_classes = self.n_secrets
        config.secret_gen.args.image_width = image_width
        config.secret_gen.args.image_height = image_height
        config.secret_gen.args.n_classes = self.n_secrets
        config.secret_disc.args.n_classes = self.n_secrets + 1
        # config.label_classifier.args.n_classes = self.n_secrets
        # config.secret_classifier.args.n_classes = self.n_labels

        self.loss_funcs = {'filter_gen_distortion': config.loss.filter_dist_loss(), 'filter_gen_entropy': HLoss(),
                           'filter_gen_adversarial': torch.nn.CrossEntropyLoss(
                               label_smoothing=config.filter_gen_label_smoothing),
                           'filter_disc': torch.nn.CrossEntropyLoss(
                               label_smoothing=config.filter_disc_label_smoothing),
                           'secret_gen_distortion': config.loss.filter_dist_loss(), 'secret_gen_entropy': HLoss(),
                           'secret_gen_adversarial': torch.nn.CrossEntropyLoss(
                               label_smoothing=config.secret_gen_label_smoothing),
                           'secret_disc': torch.nn.CrossEntropyLoss(
                               label_smoothing=config.secret_disc_label_smoothing)}

        self.filter_gen = create_model_from_config(config.filter_gen).to(device)
        self.filter_disc = create_model_from_config(config.filter_disc).to(device)
        self.secret_gen = create_model_from_config(config.secret_gen).to(device)
        self.secret_disc = create_model_from_config(config.secret_disc).to(device)

        self.lr = config.lr
        betas = config.betas
        self.optimizers = {
            'filter_gen': torch.optim.Adam(self.filter_gen.parameters(), self.lr['filter_gen'], betas['filter_gen']),
            'filter_disc': torch.optim.Adam(self.filter_disc.parameters(), self.lr['filter_disc'],
                                            betas['filter_disc']),
            'secret_gen': torch.optim.Adam(self.secret_gen.parameters(), self.lr['secret_gen'], betas['secret_gen']),
            'secret_disc': torch.optim.Adam(self.secret_disc.parameters(), self.lr['secret_disc'], betas['secret_disc'])
        }
        self.disc_optimizers = {key:self.optimizers[key] for key in ['filter_disc', 'secret_disc']}
        self.gen_optimizers = {key: self.optimizers[key] for key in ['filter_gen', 'secret_gen']}

        self.loss_config = config.loss
        self.generate_both_secrets = config.generate_both_secrets
        self.sampling_rate = config.audio2mel.args.sampling_rate
        self.device = device

        self.means = None
        self.stds = None

    def pad(self, data, target_length):
        lth = data.shape[-1]
        p = target_length - lth
        output = torch.nn.functional.pad(data, (0, p), "constant", 0)
        return output

    def preprocess(self, audio):
        # audio = torch.from_numpy(librosa.resample(y=audio.numpy(), orig_sr=16000, target_sr=22050))
        # audio = self.pad(audio, 57500)
        mels = self.audio2mel(audio).detach()  # mels: (bsz, n_mels, frames)
        mels, self.means, self.stds = preprocess_spectrograms(mels)
        mels = mels.to(self.device)  # mels: (bsz, 1, n_mels, frames)
        # print(mels.shape)
        # assert mels.dim() == 4
        return mels

    def postprocess(self, audio, mel, filtered_mel, fake_mel_male, fake_mel_female):

        # unnormalize
        # filtered_mel = (torch.squeeze(filtered_mel.cpu(), 1) * 3 * std + mean)
        # fake_mel_male = (torch.squeeze(fake_mel_male.cpu(), 1) * 3 * std + mean)
        # fake_mel_female = (torch.squeeze(fake_mel_female.cpu(), 1) * 3 * std + mean)
        # mel = torch.squeeze(mel.cpu() * 3 * std + mean)

        print(mel.shape, filtered_mel.shape, fake_mel_male.shape, fake_mel_female.shape)

        if self.stds and self.means:
            mel = mel.cpu() * 3 * self.stds + self.means
            filtered_mel = (filtered_mel.cpu() * 3 * self.stds) + self.means
            fake_mel_male = (fake_mel_male.cpu() * 3 * self.stds) + self.means
            fake_mel_female = (fake_mel_female.cpu() * 3 * self.stds) + self.means

        audio = audio.squeeze().cpu()
        a2m2_audio = self.mel2audio(mel.squeeze().cpu())
        filtered_audio = self.mel2audio(filtered_mel.squeeze().cpu())
        audio_male = self.mel2audio(fake_mel_male.squeeze().cpu())
        audio_female = self.mel2audio(fake_mel_female.squeeze().cpu())

        a2m2_audio = a2m2_audio.squeeze()
        filtered_audio = filtered_audio.squeeze()
        audio_male = audio_male.squeeze()
        audio_female = audio_female.squeeze()

        # resample back to original sample_rate
        # filtered_audio = a2m2_audio.clone()
        # a2m2_audio = torch.from_numpy(librosa.resample(y=a2m2_audio.numpy(), orig_sr=22050, target_sr=16000))
        # filtered_audio = torch.from_numpy(librosa.resample(y=filtered_audio.numpy(), orig_sr=22050, target_sr=16000))
        # audio_male = torch.from_numpy(librosa.resample(y=audio_male.numpy(), orig_sr=22050, target_sr=16000))
        # audio_female = torch.from_numpy(librosa.resample(y=audio_female.numpy(), orig_sr=22050, target_sr=16000))

        print(audio.shape, a2m2_audio.shape, filtered_audio.shape, audio_male.shape, audio_female.shape)
        # if a2m2_audio.shape[-1] > audio.shape[-1]:
        #     cut_off = audio.shape[-1]  # 64000
        #     a2m2_audio = a2m2_audio[..., :cut_off]  # .squeeze(dim=0)
        #     filtered_audio = filtered_audio[..., :cut_off]  # .squeeze(dim=0)
        #     audio_male = audio_male[..., :cut_off]  # .squeeze(dim=0)
        #     audio_female = audio_female[..., :cut_off]  # .squeeze(dim=0)
        #     print(audio.shape, a2m2_audio.shape, filtered_audio.shape, audio_male.shape, audio_female.shape)

        return audio, a2m2_audio, filtered_audio, audio_male, audio_female

    def _filter_gen_forward_pass(self, mels, secrets):
        mel_encodings = self.audio_encoder(mels)
        print('audio_encoder')
        GPUtil.showUtilization()
        filter_z = torch.randn(mels.shape[0], self.filter_gen.noise_dim).to(mels.device)
        print('torch.randn')
        GPUtil.showUtilization()
        filtered_mels = self.filter_gen(mels.unsqueeze(dim=1), filter_z, secrets.long())  # (bsz, 1, n_mels, frames)
        print('filter_gen forward')
        GPUtil.showUtilization()
        filtered_secret_preds_gen = self.filter_disc(filtered_mels, frozen=True)  # (bsz, n_secret)
        print('filter disc forward')
        GPUtil.showUtilization()
        filtered_mel_encodings = self.audio_encoder(filtered_mels.squeeze(dim=1))
        print('filtered audio_encoder')
        GPUtil.showUtilization()
        return {'filtered_mel': filtered_mels, 'filtered_secret_score': filtered_secret_preds_gen,
                'mel_encodings': mel_encodings, 'filtered_mel_encodings': filtered_mel_encodings}

    def _secret_gen_forward_pass(self, mels, filtered_mel):
        secret_z = torch.randn(mels.shape[0], self.secret_gen.noise_dim).to(mels.device)
        fake_secret_gen = torch.randint(0, 1, mels.shape[0:1]).to(mels.device)  # (bsz,)
        fake_mel = self.secret_gen(filtered_mel.detach(), secret_z, fake_secret_gen)  # (bsz, 1, n_mels, frames)
        fake_secret_preds_gen = self.secret_disc(fake_mel, frozen=True)  # (bsz, n_secrets + 1)
        fake_mel_encodings = self.audio_encoder(fake_mel.squeeze(dim=1))

        secret_gen_output = {'fake_secret': fake_secret_gen, 'faked_mel': fake_mel,
                             'fake_secret_score': fake_secret_preds_gen, 'fake_mel_encodings': fake_mel_encodings}

        if self.generate_both_secrets:
            # (bsz, 1, n_mels, frames)
            alt_fake_secret_gen = 1 - fake_secret_gen
            alt_fake_mel = self.secret_gen(filtered_mel.detach(), secret_z, alt_fake_secret_gen, frozen=True)
            # (bsz, n_secrets + 1)
            alt_fake_secret_preds_gen = self.secret_disc(alt_fake_mel, frozen=True)
            alt_fake_mel_encodings = self.audio_encoder(alt_fake_mel.squeeze(dim=1))
            secret_gen_output.update(
                {'alt_fake_secret': alt_fake_secret_gen, 'alt_faked_mel': alt_fake_mel,
                 'alt_fake_secret_score': alt_fake_secret_preds_gen, 'alt_fake_mel_encodings': alt_fake_mel_encodings})

        return secret_gen_output

    def _filter_disc_forward_pass(self, mels, filtered_mels):
        filtered_secret_preds_disc = self.filter_disc(filtered_mels.detach())  # (bsz, n_secret)
        unfiltered_secret_preds_disc = self.filter_disc(mels.unsqueeze(dim=1).detach(), frozen=True)  # (bsz, n_secret)
        return {'filtered_secret_score': filtered_secret_preds_disc,
                'unfiltered_secret_score': unfiltered_secret_preds_disc}

    def _secret_disc_forward_pass(self, mels, fake_mel):
        fake_secret_preds_disc = self.secret_disc(fake_mel.detach().clone())  # (bsz, n_secrets + 1)
        real_secret_preds_disc = self.secret_disc(mels.unsqueeze(dim=1))  # (bsz, n_secrets + 1)
        fake_secret_disc = 2 * torch.ones(mels.size(0), requires_grad=False, dtype=torch.int64).to(
            mels.device)  # (bsz,)
        return {'fake_secret_score': fake_secret_preds_disc, 'real_secret_score': real_secret_preds_disc,
                'fake_secret': fake_secret_disc}

    def forward_pass(self, audio, secrets, labels):
        mels = self.preprocess(audio)
        print('forward_pass')
        GPUtil.showUtilization()
        filter_gen_output = self._filter_gen_forward_pass(mels, secrets)
        print('filter_gen_forward_pass')
        GPUtil.showUtilization()
        secret_gen_output = self._secret_gen_forward_pass(mels, filter_gen_output['filtered_mel'])
        print('secret_gen_forward_pass')
        GPUtil.showUtilization()
        filter_disc_output = self._filter_disc_forward_pass(mels, filter_gen_output['filtered_mel'])
        print('filter_disc_forward_pass')
        GPUtil.showUtilization()
        secret_disc_output = self._secret_disc_forward_pass(mels, secret_gen_output['faked_mel'])
        print('secret_disc_forward_pass')
        GPUtil.showUtilization()

        # label_preds = self.label_classifier(secret_gen_output['faked_mel'])
        # secret_preds = self.secret_classifier(secret_gen_output['faked_mel'])
        # secret_disc_output.update({'label_score': label_preds, 'secret_score': secret_preds})
        losses = compute_losses(self.loss_funcs, mels, secrets, filter_gen_output, filter_disc_output,
                                secret_gen_output, secret_disc_output, self.loss_config)
        print('compute_losses')
        GPUtil.showUtilization()
        batch_metrics = compute_metrics(mels, secrets, labels, filter_gen_output, filter_disc_output, secret_gen_output,
                                        secret_disc_output, losses, self.loss_funcs)
        print('compute_metrics')
        GPUtil.showUtilization()
        return batch_metrics, losses

    def generate_sample(self, audio, secret, label, id, epoch):

        mel = self.preprocess(audio)

        # filter_gen
        filter_z = torch.randn(mel.shape[0], self.filter_gen.noise_dim).to(self.device)
        filtered_mel = self.filter_gen(mel.unsqueeze(dim=1), filter_z, secret.long())

        # secret_gen
        secret_z = torch.randn(mel.shape[0], self.secret_gen.noise_dim).to(self.device)
        fake_secret_male = torch.ones(mel.shape[0], requires_grad=False, dtype=torch.int64).to(self.device)
        fake_secret_female = torch.zeros(mel.shape[0], requires_grad=False, dtype=torch.int64).to(self.device)
        fake_mel_male = self.secret_gen(filtered_mel, secret_z, fake_secret_male)
        fake_mel_female = self.secret_gen(filtered_mel, secret_z, fake_secret_female)

        # secret disc
        pred_label_male = torch.argmax(self.secret_disc(fake_mel_male.detach().clone()).data, 1)
        pred_label_female = torch.argmax(self.secret_disc(fake_mel_female.detach().clone()).data, 1)

        audio, a2m2_audio, filtered_audio, audio_male, audio_female = self.postprocess(audio, mel, filtered_mel,
                                                                                       fake_mel_male, fake_mel_female)
        print(audio.shape, a2m2_audio.shape, filtered_audio.shape, audio_male.shape, audio_female.shape)

        output = self._build_output_dict(id, label, epoch, audio, pred_label_male, pred_label_female, filtered_audio,
                                         a2m2_audio, audio_male, audio_female)

        [print(v.shape) for k, v in output.items()]
        return output

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

    def _build_output_dict(self, id, label, epoch, audio, pred_label_male, pred_label_female, filtered_audio,
                           a2m2_audio, audio_male, audio_female):
        speaker_digit_str = f'speaker_{id.item()}_label_{label.item()}'
        speaker_digit_epoch_str = speaker_digit_str + f'_epoch_{epoch}'

        print(pred_label_male.shape, pred_label_female.shape)

        def build_str(gnd, lbl): return speaker_digit_epoch_str + f'_sampled_gender_{gnd}_predicted_label_{lbl}.wav'

        filtered_audio_file = speaker_digit_epoch_str + '_filtered.wav'
        a2m2a_file = speaker_digit_epoch_str + '_a2m2a.wav'
        male_audio_file = build_str('male', pred_label_male.item())
        female_audio_file = build_str('female', pred_label_female.item())
        original_audio_file = speaker_digit_str + '.wav'

        return {original_audio_file: audio,
                filtered_audio_file: filtered_audio,
                a2m2a_file: a2m2_audio,
                male_audio_file: audio_male,
                female_audio_file: audio_female}
