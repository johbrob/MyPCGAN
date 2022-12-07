from sklearn.metrics import accuracy_score
import numpy as np
import torch


def compute_metrics(secret, label, filter_gen_output, filter_disc_output, secret_gen_output,
                    secret_disc_output, losses):
    filter_gen_metrics = {'distortion_loss': losses['filter_gen']['distortion'].detach().cpu().numpy(),
                          'adversarial_loss': losses['filter_gen']['adversarial'].detach().cpu().numpy(),
                          'combined_loss': losses['filter_gen']['final'].detach().cpu().numpy()}

    secret_gen_metrics = {'distortion_loss': losses['secret_gen']['distortion'].detach().cpu().numpy(),
                          'adversarial_loss': losses['secret_gen']['adversarial'].detach().cpu().numpy(),
                          'combined_loss': losses['secret_gen']['final'].detach().cpu().numpy()}

    # filter_disc
    filtered_secret_preds_disc = torch.argmax(filter_disc_output['filtered_secret_score'], 1).cpu().numpy()
    filtered_secret_accuracy_disc = np.array(accuracy_score(secret.cpu().numpy(), filtered_secret_preds_disc))
    filter_disc_metrics = {'loss': losses['filter_disc']['final'].detach().cpu().numpy(),
                           'accuracy': filtered_secret_accuracy_disc}

    # secret_disc
    # print(secret_disc_output['fake_secret_score'].shape)
    fake_secret_preds_disc = torch.argmax(secret_disc_output['fake_secret_score'], 1).cpu().numpy()
    fake_secret_label_accuracy_disc = np.array(accuracy_score(secret_disc_output['fake_secret'].cpu().numpy(),
                                                              fake_secret_preds_disc))
    real_secret_preds_disc = torch.argmax(secret_disc_output['real_secret_score'], 1).cpu().numpy()
    real_secret_label_accuracy_disc = np.array(accuracy_score(secret.cpu().numpy(), real_secret_preds_disc))
    generated_secret_accuracy_disc = np.array(accuracy_score(secret_gen_output['fake_secret'].cpu().numpy(),
                                                             fake_secret_preds_disc))

    secret_disc_metrics = {'real_loss': losses['secret_disc']['real'].detach().cpu().numpy(),
                           'fake_loss': losses['secret_disc']['fake'].detach().cpu().numpy(),
                           'combined_loss': losses['secret_disc']['final'].detach().cpu().numpy(),
                           'fake_accuracy': fake_secret_label_accuracy_disc,
                           'real_accuracy': real_secret_label_accuracy_disc,
                           'generated_accuracy': generated_secret_accuracy_disc}

    # label prediction
    label_preds_disc = torch.argmax(secret_disc_output['label_score'].data, 1).cpu().numpy()
    label_accuracy_disc = np.array(accuracy_score(label.cpu().numpy(), label_preds_disc))
    label_prediction_metrics = {'accuracy': label_accuracy_disc}

    secret_preds_disc = torch.argmax(secret_disc_output['secret_score'].data, 1).cpu().numpy()
    real_secret_accuracy_disc = np.array(accuracy_score(secret.cpu().numpy(), secret_preds_disc))
    fake_secret_accuracy_disc = np.array(
        accuracy_score(secret_gen_output['fake_secret'].cpu().numpy(), secret_preds_disc))
    secret_prediction_metrics = {'real_secret_accuracy': real_secret_accuracy_disc,
                                 'fake_secret_accuracy': fake_secret_accuracy_disc}

    return {'filter_gen': filter_gen_metrics, 'secret_gen': secret_gen_metrics,
            'filter_disc': filter_disc_metrics, 'secret_disc': secret_disc_metrics,
            'alexNet_label_pred': label_prediction_metrics, 'alexNet_secret_pred': secret_prediction_metrics}


def compile_metrics(metrics):
    return {group_name + '/' + name: metric for group_name, metric_dict in metrics.items() for name, metric in
            metric_dict.items()}


def aggregate_metrics(batch_metrics, metrics):
    for k, v in batch_metrics.items():
        if k not in metrics:
            metrics[k] = []
        if isinstance(v, np.ndarray):
            metrics[k].append(v.item())
        else:
            raise NotImplementedError(f'Unexpected format of metric: {k}, {v}')

    return metrics
