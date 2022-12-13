import torch


class LossConfig:
    def __init__(self, gamma=100, use_entropy_loss=False):
        self.gamma = gamma
        self.use_entropy_loss = use_entropy_loss


class HLoss(torch.nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = torch.nn.functional.softmax(x, dim=1) * torch.nn.functional.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b


def _compute_filter_gen_loss(loss_funcs, spectrograms, secret, filter_gen_output, gamma, entropy_loss):
    ones = torch.ones(secret.shape, requires_grad=True, dtype=torch.float32).to(spectrograms.device)
    target = ones - secret.float()
    target = target.view(target.size(0))
    distortion_loss = loss_funcs['distortion'](filter_gen_output['filtered_mel'], spectrograms)

    if entropy_loss or True:
        adversary_loss = loss_funcs['entropy'](filter_gen_output['filtered_secret_score'])
    else:
        adversary_loss = loss_funcs['adversarial'](filter_gen_output['filtered_secret_score'], target.long())

    final_loss = adversary_loss + gamma * torch.pow(torch.relu(distortion_loss - 1e-3), 2)

    return {'distortion': distortion_loss, 'adversarial': adversary_loss, 'final': final_loss}


def _compute_secret_gen_loss(loss_func, spectrograms, secret_gen_output, gamma):
    distortion_loss = loss_func['distortion'](secret_gen_output['faked_mel'], spectrograms)
    adversary_loss = loss_func['adversarial'](secret_gen_output['fake_secret_score'], secret_gen_output['fake_secret'])
    final_loss = adversary_loss + gamma * torch.pow(torch.relu(distortion_loss - 1e-3), 2)

    return {'distortion': distortion_loss, 'adversarial': adversary_loss, 'final': final_loss,
            'alt_distortion': distortion_loss, 'alt_adversarial': adversary_loss, 'alt_final': final_loss}


def _compute_filter_disc_loss(loss_func, secret, filter_disc_output):
    return {'final': loss_func['adversarial'](filter_disc_output['filtered_secret_score'], secret.long()),
            'unfiltered_score_loss': loss_func['adversarial'](filter_disc_output['unfiltered_secret_score'].detach(), secret.long())}



def _compute_secret_disc_loss(loss_func, secret, secret_disc_output):
    real_loss = loss_func['adversarial_rf'](secret_disc_output['real_secret_score'],
                                            secret.long().to(secret_disc_output['fake_secret_score'].device)).to(
        secret_disc_output['fake_secret_score'].device)
    fake_loss = loss_func['adversarial'](secret_disc_output['fake_secret_score'], secret_disc_output['fake_secret']).to(
        secret_disc_output['fake_secret_score'].device)
    average_loss = (real_loss + fake_loss) / 2

    return {'real': real_loss, 'fake': fake_loss, 'final': average_loss}


def compute_losses(loss_funcs, spectrograms, secret, filter_gen_output, filter_disc_output, secret_gen_output,
                   secret_disc_output, gamma, filter_gen_entropy_loss):
    losses = {}
    losses['filter_gen'] = _compute_filter_gen_loss(loss_funcs, spectrograms, secret, filter_gen_output, gamma,
                                                    filter_gen_entropy_loss)
    losses['secret_gen'] = _compute_secret_gen_loss(loss_funcs, spectrograms, secret_gen_output, gamma)
    losses['filter_disc'] = _compute_filter_disc_loss(loss_funcs, secret, filter_disc_output)
    losses['secret_disc'] = _compute_secret_disc_loss(loss_funcs, secret, secret_disc_output)

    return losses
