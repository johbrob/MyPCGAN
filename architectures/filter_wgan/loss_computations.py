import torch


class LossConfig:
    def __init__(self, filter_gamma=1000, filter_epsilon=1e-3, filter_entropy_loss=False, secret_gamma=1000,
                 secret_epsilon=1e-3):
        self.filter_gamma = filter_gamma
        self.filter_epsilon = filter_epsilon
        self.filter_entropy_loss = filter_entropy_loss

        self.secret_gamma = secret_gamma
        self.secret_epsilon = secret_epsilon


def _compute_filter_gen_loss(loss_funcs, spectrograms, secret, filter_gen_output, loss_config):
    raw_distortion = loss_funcs['filter_gen_distortion'](filter_gen_output['filtered_mel_encodings'],
                                                         filter_gen_output['mel_encodings'])

    # use only first of two columns, 0 for female, 1 for male
    errD_male = torch.mean(filter_gen_output['filtered_secret_score'][(secret == 1), 0])
    errD_female = torch.mean(filter_gen_output['filtered_secret_score'][(secret == 0), 0])
    errD = errD_female - errD_male  # + gradient_penalty * 10

    distortion_loss = loss_config.filter_gamma * torch.pow(torch.relu(raw_distortion - loss_config.filter_epsilon), 2)
    final_loss = errD + distortion_loss

    return {'raw_distortion': raw_distortion, 'distortion': distortion_loss, 'adversarial': errD, 'final': final_loss}


def _compute_filter_disc_loss(secret, filter_disc_output):
    errD_male = torch.mean(filter_disc_output['filtered_secret_score'][(secret == 1), 0])
    errD_female = torch.mean(filter_disc_output['filtered_secret_score'][(secret == 0), 0])
    errD = -errD_female + errD_male  # + gradient_penalty * 10

    unfiltered_errD_male = torch.mean(filter_disc_output['unfiltered_secret_score'][(secret == 1), 0])
    unfiltered_errD_female = torch.mean(filter_disc_output['unfiltered_secret_score'][(secret == 0), 0])

    lambda_gp = 10
    unfiltered_errD = -unfiltered_errD_female + unfiltered_errD_male + lambda_gp * filter_disc_output['gradient_penalty']
    return {'final': errD, 'male_filtered_loss': errD_male, 'female_filtered_loss': errD_female,
            'unfiltered_final': unfiltered_errD,
            'male_unfiltered_loss': unfiltered_errD_male, 'female_unfiltered_loss': unfiltered_errD_female}


def compute_losses(loss_funcs, mels, secret, filter_gen_output, filter_disc_output, loss_config):
    return {'filter_gen': _compute_filter_gen_loss(loss_funcs, mels, secret, filter_gen_output, loss_config),
            'filter_disc': _compute_filter_disc_loss(secret, filter_disc_output)}
