from sklearn.metrics import accuracy_score
import numpy as np
import torch


def _compute_filter_gen_metrics(losses):
    return {'raw_distortion': losses['filter_gen']['raw_distortion'].detach().cpu().numpy(),
            'distortion_loss': losses['filter_gen']['distortion'].detach().cpu().numpy(),
            'adversarial_loss': losses['filter_gen']['adversarial'].detach().cpu().numpy(),
            'combined_loss': losses['filter_gen']['final'].detach().cpu().numpy()}


def _compute_filter_disc_metrics(losses, filter_disc_output, secret):
    # filtered_secret_preds_disc = torch.argmax(filter_disc_output['filtered_secret_score'], 1).cpu().numpy()
    # filtered_secret_accuracy_disc = np.array(accuracy_score(secret.cpu().numpy(), filtered_secret_preds_disc))
    #
    # unfiltered_secret_preds_disc = torch.argmax(filter_disc_output['unfiltered_secret_score'], 1).cpu().numpy()
    # unfiltered_secret_accuracy_disc = np.array(accuracy_score(secret.cpu().numpy(), unfiltered_secret_preds_disc))

    metrics = {'filtered_loss': losses['filter_disc']['final'].detach().cpu().numpy(),
               # 'filtered_accuracy': filtered_secret_accuracy_disc,
               'unfiltered_loss': losses['filter_disc']['unfiltered_final'].detach().cpu().numpy(),
               # 'unfiltered_accuracy': unfiltered_secret_accuracy_disc,
               }
    if len(torch.unique(secret)) > 1:
        metrics.update({
            'filtered_male': losses['filter_disc']['male_filtered_loss'].detach().cpu().numpy(),
            'filtered_female': losses['filter_disc']['female_filtered_loss'].detach().cpu().numpy(),
            'unfiltered_male': losses['filter_disc']['male_unfiltered_loss'].detach().cpu().numpy(),
            'unfiltered_female': losses['filter_disc']['female_unfiltered_loss'].detach().cpu().numpy(),
        })
    else:
        print(secret, torch.unique(secret))
    return metrics

def compute_metrics(mels, secret, label, filter_gen_output, filter_disc_output, losses, loss_funcs):
    filter_gen_metrics = _compute_filter_gen_metrics(losses)
    filter_disc_metrics = _compute_filter_disc_metrics(losses, filter_disc_output, secret)

    return {'filter_gen': filter_gen_metrics, 'filter_disc': filter_disc_metrics}
