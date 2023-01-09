from sklearn.metrics import accuracy_score
import numpy as np
import torch


def _compute_gen_metrics(losses):
    return {'fake_loss': losses['gen']['fake'].detach().cpu().numpy(),
            'secret_loss': losses['gen']['secret'].detach().cpu().numpy(),
            'combined_loss': losses['gen']['final'].detach().cpu().numpy()}


def _compute_fake_disc_metrics(losses, fake_disc_output):
    bsz = fake_disc_output['fake_mel_score'].shape[0]

    fake_mel_preds = torch.argmax(fake_disc_output['fake_mel_score'], 1).cpu().numpy()
    fake_mel_accuracy = np.array(accuracy_score(np.ones(bsz), fake_mel_preds))
    fake_mel_loss = losses['fake_disc']['fake_loss'].detach().cpu().numpy()

    mel_preds = torch.argmax(fake_disc_output['mel_score'], 1).cpu().numpy()
    mel_accuracy = np.array(accuracy_score(np.zeros(bsz), mel_preds))
    mel_loss = losses['fake_disc']['real_loss'].detach().cpu().numpy()

    return {'fake_mel_loss': fake_mel_loss, 'fake_mel_accuracy': fake_mel_accuracy,
            'real_mel_loss': mel_loss, 'real_mel_accuracy': mel_accuracy,
            'combined_loss': fake_mel_loss + mel_loss, 'combined_accuracy': (fake_mel_accuracy + mel_accuracy) / 2}


def _compute_secret_disc_metrics(losses, secret_disc_output, fake_secret, secret):
    fake_preds = torch.argmax(secret_disc_output['fake_secret_score'], 1).cpu().numpy()
    fake_accuracy = np.array(accuracy_score(fake_secret.cpu().numpy(), fake_preds))
    real_preds = torch.argmax(secret_disc_output['real_secret_score'], 1).cpu().numpy()
    real_accuracy = np.array(accuracy_score(secret.cpu().numpy(), real_preds))

    return {'real_loss': losses['secret_disc']['real'].detach().cpu().numpy(),
            'fake_loss': losses['secret_disc']['fake'].detach().cpu().numpy(),
            'combined_loss': losses['secret_disc']['final'].detach().cpu().numpy(),
            'fake_accuracy': fake_accuracy, 'real_accuracy': real_accuracy}


def compute_metrics(secret, gen_output, fake_disc_output, secret_disc_output, losses):
    gen_metrics = _compute_gen_metrics(losses)
    fake_disc_metrics = _compute_fake_disc_metrics(losses, fake_disc_output)
    secret_disc_metrics = _compute_secret_disc_metrics(losses, secret_disc_output, secret, gen_output['fake_secret'])

    return {'gen': gen_metrics, 'fake_disc': fake_disc_metrics, 'secret_disc': secret_disc_metrics}


def compile_metrics(metrics):
    metrics = {group_name + '/' + name: metric for group_name, metric_dict in metrics.items() for name, metric
               in metric_dict.items()}
    return metrics


def aggregate_metrics(batch_metrics, metrics):
    for k, v in batch_metrics.items():
        if k not in metrics:
            metrics[k] = []
        if isinstance(v, np.ndarray):
            metrics[k].append(v.item())
        elif isinstance(v, int):
            metrics[k].append(v)
        else:
            raise NotImplementedError(f'Unexpected format of metric: {k}, {v}')

    return metrics
