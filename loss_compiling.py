import torch


class LossConfig:
    def __init__(self, filter_gamma=100, filter_epsilon=1e-3, filter_entropy_loss=False, secret_gamma=100,
                 secret_epsilon=1e-3, filter_gen_label_smoothing=0, filter_disc_label_smoothing=0,
                 secret_gen_label_smoothing=0, secret_disc_label_smoothing=0):
        self.filter_gen_label_smoothing = filter_gen_label_smoothing
        self.filter_disc_label_smoothing = filter_disc_label_smoothing
        self.secret_gen_label_smoothing = secret_gen_label_smoothing
        self.secret_disc_label_smoothing = secret_disc_label_smoothing

        self.filter_gamma = filter_gamma
        self.filter_epsilon = filter_epsilon
        self.filter_entropy_loss = filter_entropy_loss

        self.secret_gamma = secret_gamma
        self.secret_epsilon = secret_epsilon


class HLoss(torch.nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = torch.nn.functional.softmax(x, dim=1) * torch.nn.functional.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b


def _compute_filter_gen_loss(loss_funcs, spectrograms, secret, filter_gen_output, loss_config):
    ones = torch.ones(secret.shape, requires_grad=True, dtype=torch.float32).to(spectrograms.device)
    target = ones - secret.float()
    target = target.view(target.size(0))
    distortion_loss = loss_funcs['filter_gen_distortion'](filter_gen_output['filtered_mel'], spectrograms)

    if loss_config.use_entropy_loss or True:
        adversary_loss = loss_funcs['filter_gen_entropy'](filter_gen_output['filtered_secret_score'])
    else:
        adversary_loss = loss_funcs['filter_gen_adversarial'](filter_gen_output['filtered_secret_score'], target.long())

    final_loss = adversary_loss + \
                 loss_config.filter_gamma * torch.pow(torch.relu(distortion_loss - loss_config.filter_epsilon), 2)

    return {'distortion': distortion_loss, 'adversarial': adversary_loss, 'final': final_loss}


def _compute_secret_gen_loss(loss_func, spectrograms, secret_gen_output, loss_config):
    distortion_loss = loss_func['secret_gen_distortion'](secret_gen_output['faked_mel'], spectrograms)
    adversary_loss = loss_func['secret_gen_adversarial'](secret_gen_output['fake_secret_score'], secret_gen_output['fake_secret'])
    final_loss = adversary_loss + \
                 loss_config.secret_gamma * torch.pow(torch.relu(distortion_loss - loss_config.secret_epsilon), 2)

    return {'distortion': distortion_loss, 'adversarial': adversary_loss, 'final': final_loss,
            'alt_distortion': distortion_loss, 'alt_adversarial': adversary_loss, 'alt_final': final_loss}


def _compute_filter_disc_loss(loss_func, secret, filter_disc_output):
    return {'final': loss_func['filter_disc_adversarial'](filter_disc_output['filtered_secret_score'], secret.long()),
            'unfiltered_score_loss': loss_func['filter_disc_adversarial'](filter_disc_output['unfiltered_secret_score'].detach(),
                                                              secret.long())}


def _compute_secret_disc_loss(loss_func, secret, secret_disc_output):
    real_loss = loss_func['secret_adversarial_rf'](secret_disc_output['real_secret_score'],
                                            secret.long().to(secret_disc_output['fake_secret_score'].device)).to(
        secret_disc_output['fake_secret_score'].device)
    fake_loss = loss_func['secret_adversarial'](secret_disc_output['fake_secret_score'], secret_disc_output['fake_secret']).to(
        secret_disc_output['fake_secret_score'].device)
    average_loss = (real_loss + fake_loss) / 2

    return {'real': real_loss, 'fake': fake_loss, 'final': average_loss}


def compute_losses(loss_funcs, spectrograms, secret, filter_gen_output, filter_disc_output, secret_gen_output,
                   secret_disc_output, loss_config):
    losses = {}
    losses['filter_gen'] = _compute_filter_gen_loss(loss_funcs, spectrograms, secret, filter_gen_output, loss_config)
    losses['secret_gen'] = _compute_secret_gen_loss(loss_funcs, spectrograms, secret_gen_output, loss_config)
    losses['filter_disc'] = _compute_filter_disc_loss(loss_funcs, secret, filter_disc_output)
    losses['secret_disc'] = _compute_secret_disc_loss(loss_funcs, secret, secret_disc_output)

    return losses
