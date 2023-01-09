from architectures.alt_gen.metrics_compiling import compute_metrics
from architectures.utils import create_model_from_config
from architectures.alt_gen import loss_compiling
from utils import Mode
import torch
import tqdm
import os


class OneStepGanConfig:
    def __init__(self, gen_config, fake_disc_config, secret_disc_config, lr=None, betas=None,
                 generate_both_secrets=True, fake_disc_label_smoothing=0, secret_disc_label_smoothing=0):
        self.architecture = OneStepGAN

        self.gen = gen_config
        self.fake_disc = fake_disc_config
        self.secret_disc = secret_disc_config

        if lr is None:
            self.lr = {'gen': 0.0001, 'fake_disc': 0.0004, 'secret_disc': 0.0004}
        else:
            self.lr = lr

        if betas is None:
            self.betas = {'gen': (0.5, 0.999), 'fake_disc': (0.5, 0.999), 'secret_disc': (0.5, 0.999)}
        else:
            self.betas = betas

        self.generate_both_secrets = generate_both_secrets
        self.fake_disc_label_smoothing = fake_disc_label_smoothing
        self.secret_disc_label_smoothing = secret_disc_label_smoothing


class OneStepGAN:

    def __init__(self, config, device):

        config.gen.args.image_width = config.image_width
        config.gen.args.image_height = config.image_height
        config.gen.args.n_classes = config.n_genders
        config.fake_disc.args.n_classes = config.n_genders
        config.secret_disc.args.n_classes = config.n_genders

        # models
        self.gen = create_model_from_config(config.gen).to(device)
        self.fake_disc = create_model_from_config(config.fake_disc).to(device)
        self.secret_disc = create_model_from_config(config.secret_disc).to(device)

        # optimizers
        self.lr = config.lr
        betas = config.betas
        self.optimizers = {
            'gen': torch.optim.Adam(self.gen.parameters(), self.lr['gen'], betas['gen']),
            'fake_disc': torch.optim.Adam(self.fake_disc.parameters(), self.lr['fake_disc'], betas['fake_disc']),
            'secret_disc': torch.optim.Adam(self.secret_disc.parameters(), self.lr['secret_disc'], betas['secret_disc'])
        }

        self.loss_funcs = {
            'gen_fake_ce': torch.nn.CrossEntropyLoss(), 'gen_secret_ce': torch.nn.CrossEntropyLoss(),
            'fake_disc': torch.nn.CrossEntropyLoss(label_smoothing=config.fake_disc_label_smoothing),
            'secret_disc': torch.nn.CrossEntropyLoss(label_smoothing=config.secret_disc_label_smoothing)
        }

        self.generate_both_secrets = config.generate_both_secrets

    def _gen_forward_pass(self, mels):
        z = torch.randn(mels.shape[0], self.gen.noise_dim).to(mels.device)
        fake_secret = torch.randint(0, 1, mels.shape[0:1]).to(mels.device)  # (bsz,)
        fake_mels = self.gen(mels, z, fake_secret)  # (bsz, 1, n_mels, frames)

        fake_mels_score = self.fake_disc(fake_mels, frozen=True)  # (bsz, n_secret)
        secret_score = self.secret_disc(fake_mels, frozen=True)  # (bsz, n_secret)

        if self.generate_both_secrets:
            alt_fake_mels = self.gen(mels, z, 1 - fake_secret, frozen=True)  # (bsz, 1, n_mels, frames)
            alt_fake_mels_score = self.fake_disc(alt_fake_mels, frozen=True)  # (bsz, n_secrets + 1)
            alt_secret_score = self.secret_disc(alt_fake_mels, frozen=True)  # (bsz, n_secrets + 1)
        else:
            alt_fake_mels, alt_fake_mels_score, alt_secret_score = None, None, None

        return {'fake_mel': fake_mels, 'fake_secret': fake_secret, 'fake_mel_score': fake_mels_score,
                'secret_score': secret_score, 'alt_fake_mel': alt_fake_mels, 'alt_fake_mel_score': alt_fake_mels_score,
                'alt_secret_score': alt_secret_score}

    def _fake_disc_forward_pass(self, mels, fake_mels, alt_fake_mels):
        mel_score = self.fake_disc(mels.detach())  # (bsz, n_secret)
        fake_mel_score = self.fake_disc(fake_mels.detach())  # (bsz, n_secret)

        if self.generate_both_secrets:
            alt_fake_mel_score = self.fake_disc(alt_fake_mels.detach())  # (bsz, n_secret)
        else:
            alt_fake_mel_score = None

        return {'mel_score': mel_score, 'fake_mel_score': fake_mel_score, 'alt_fake_mel_score': alt_fake_mel_score}

    def _secret_disc_forward_pass(self, mels, fake_mel, alt_fake_mel):
        real_secret_score = self.secret_disc(mels)  # (bsz, n_secrets + 1)
        fake_secret_score = self.secret_disc(fake_mel.detach().clone())  # (bsz, n_secrets + 1)

        if self.generate_both_secrets:
            alt_fake_secret_score = self.secret_disc(alt_fake_mel.detach().clone())
        else:
            alt_fake_secret_score = None

        return {'fake_secret_score': fake_secret_score, 'real_secret_score': real_secret_score,
                'alt_fake_secret_score': alt_fake_secret_score}

    def forward_pass(self, mels, secrets):
        assert mels.dim() == 4

        gen_output = self._gen_forward_pass(mels)
        fake_disc_output = self._fake_disc_forward_pass(mels, gen_output['fake_mel'], gen_output['alt_fake_mel'])
        secret_disc_output = self._secret_disc_forward_pass(mels, gen_output['fake_mel'], gen_output['alt_fake_mel'])

        losses = loss_compiling.compute_losses(self.loss_funcs, mels, secrets, gen_output, fake_disc_output,
                                               secret_disc_output, self.generate_both_secrets)
        batch_metrics = compute_metrics(secrets, gen_output, fake_disc_output, secret_disc_output, losses)

        return batch_metrics, losses

    def set_mode(self, mode):
        if mode == Mode.TRAIN:
            self.gen.train()
            self.fake_disc.train()
            self.secret_disc.train()
        elif mode == Mode.EVAL:
            self.gen.eval()
            self.fake_disc.eval()
            self.secret_disc.eval()

    def generate_sample(self, audio, id, label, secret, mels, stds, means, audio_mel_converter, epoch, sampling_rate,
                        save_dir, device):

        speaker_str = f'speaker_{id.item()}_label_{label.item()}_secret_{secret.item()}'
        speaker_epoch_str = speaker_str + f'_epoch_{epoch}'
        build_str = lambda secret, digit: speaker_epoch_str + f'_sampled_secret_{secret}_predicted_secret_{label}.wav'

        # original audio
        original_audio_file = os.path.join(save_dir, speaker_str + '.wav')

        # audio passed through the pipeline, converted to mel and then back (a2m2a = audio2mel2audio)
        fake_a2m2a_pred = torch.argmax(self.fake_disc(mels), 1).detach().cpu().numpy().item()
        secret_a2m2a_pred = torch.argmax(self.secret_disc(mels), 1).detach().cpu().numpy().item()
        a2m2a_mels = torch.squeeze(mels.to(device) * 3 * stds.to(device) + means.to(device))
        audio2mel2audio = audio_mel_converter.mel2audio(a2m2a_mels.squeeze().detach().cpu())
        audio2mel2audio_file = os.path.join(
            save_dir, speaker_str + f'_converted_fake_pred_{fake_a2m2a_pred}_secret_pred_{secret_a2m2a_pred}.wav')

        # audio generated conditioned on a specific secret
        z = torch.randn(mels.shape[0], self.gen.noise_dim).to(mels.device)

        # female
        fake_secret_zero = torch.zeros(mels.shape[0:1]).to(device)
        fake_mels_zero = self.gen(mels, z, fake_secret_zero)  # (bsz, 1, n_mels, frames)
        unnormalized_zero = (torch.squeeze(fake_mels_zero, 1) * 3 * stds + means).to(device)
        fake_zero_audio = audio_mel_converter.mel2audio(unnormalized_zero.squeeze().detach().cpu())
        fake_pred = torch.argmax(self.fake_disc(mels), 1).detach().cpu().numpy().item()
        secret_pred = torch.argmax(self.secret_disc(mels), 1).detach().cpu().numpy().item()

        # male
        fake_secret_one = torch.ones(mels.shape[0:1]).to(device)
        fake_mels_one = self.gen(mels, z, fake_secret_one)  # (bsz, 1, n_mels, frames)
        unnormalized_one = (torch.squeeze(fake_mels_one, 1) * 3 * stds + means).to(device)
        fake_one_audio = audio_mel_converter.mel2audio(unnormalized_one.squeeze().detach().cpu())

        build_str = lambda secret,
                           digit: speaker_digit_epoch_str + f'_sampled_secret_{secret}_predicted_secret_{label}.wav'


filtered_audio_file = os.path.join(save_dir, speaker_digit_epoch_str + '_filtered.wav')
male_audio_file = os.path.join(save_dir, build_str('male', pred_label_male.item()))
female_audio_file = os.path.join(save_dir, build_str('female', pred_label_female.item()))

save_audio_file(filtered_audio_file, sampling_rate, filtered_audio.squeeze().detach().cpu())
save_audio_file(male_audio_file, sampling_rate, audio_male.squeeze().detach().cpu())
save_audio_file(female_audio_file, sampling_rate, audio_female.squeeze().detach().cpu())
save_audio_file(original_audio_file, sampling_rate, original_audio.squeeze().detach().cpu())

return {original_audio_file: audio.squeeze().detach().cpu(),
        audio2mel2audio_file: audio2mel2audio,
        fake_secret_zero_files: fake_secret_zero_files,
        fake_secret_one_file: fake_secret_one_audio}
