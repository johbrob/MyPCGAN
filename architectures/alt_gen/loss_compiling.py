import torch


def _compute_gen_loss(loss_funcs, mel, secret, gen_output, use_both_secrets):
    # loss for not fooling fake_discriminator
    target = torch.zeros(secret.shape, dtype=torch.int64).to(mel.device)
    fake_loss = loss_funcs['gen_fake_ce'](gen_output['fake_mel_score'], target)

    # loss for secret_discriminator not predicting the conditioned secret
    secret_loss = loss_funcs['gen_secret_ce'](gen_output['secret_score'], gen_output['fake_secret'])

    if use_both_secrets:
        alt_fake_loss = loss_funcs['gen_fake_ce'](gen_output['alt_fake_mel_score'], target)
        alt_secret_loss = loss_funcs['gen_secret_ce'](gen_output['alt_secret_score'], gen_output['fake_secret'])

        fake_loss += alt_fake_loss
        secret_loss += alt_secret_loss

    final_loss = fake_loss + secret_loss

    return {'fake': fake_loss, 'secret': secret_loss, 'final': final_loss}


def _compute_fake_disc_loss(loss_func, fake_disc_output, use_both_secrets):
    shape = fake_disc_output['fake_mel_score'].shape
    device = fake_disc_output['fake_mel_score'].device
    zeros = torch.zeros(shape, requires_grad=True, dtype=torch.float32).to(device)
    ones = torch.ones(shape, requires_grad=True, dtype=torch.float32).to(device)

    real_loss = loss_func['fake_disc'](fake_disc_output['mel_score'].detach(), ones)
    fake_loss = loss_func['fake_disc'](fake_disc_output['fake_mel_score'].detach(), zeros)

    if use_both_secrets:
        alt_fake_loss = loss_func['fake_disc'](fake_disc_output['alt_fake_mel_score'].detach(), zeros)
        fake_loss = (fake_loss + alt_fake_loss) / 2

    return {'real_loss': real_loss, 'fake_loss': fake_loss, 'final': real_loss + fake_loss}


def _compute_secret_disc_loss(loss_func, secret, fake_secret, secret_disc_output, use_both_secrets):
    device = secret_disc_output['fake_secret_score'].device
    real_loss = loss_func['secret_disc'](secret_disc_output['real_secret_score'], secret.long().to(device)).to(device)
    fake_loss = loss_func['secret_disc'](secret_disc_output['fake_secret_score'], fake_secret).to(device)

    if use_both_secrets:
        alt_fake_loss = loss_func['secret_disc'](secret_disc_output['alt_fake_secret_score'], 1 - fake_secret).to(
            device)
        fake_loss = (fake_loss + alt_fake_loss) / 2
    average_loss = (real_loss + fake_loss) / 2

    return {'real': real_loss, 'fake': fake_loss, 'final': average_loss}


def compute_losses(loss_funcs, mels, secret, gen_output, fake_disc_output, secret_disc_output, generate_both_secrets):
    losses = {}
    losses['gen'] = _compute_gen_loss(loss_funcs, mels, secret, gen_output, generate_both_secrets)
    losses['fake_disc'] = _compute_fake_disc_loss(loss_funcs, fake_disc_output, generate_both_secrets)
    losses['secret_disc'] = _compute_secret_disc_loss(loss_funcs, secret, gen_output['fake_secret'], secret_disc_output,
                                                      generate_both_secrets)

    return losses
