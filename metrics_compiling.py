from sklearn.metrics import accuracy_score
import numpy as np
import torch


def _compute_filter_gen_metrics(losses):
    return {'distortion_loss': losses['filter_gen']['distortion'].detach().cpu().numpy(),
            'adversarial_loss': losses['filter_gen']['adversarial'].detach().cpu().numpy(),
            'combined_loss': losses['filter_gen']['final'].detach().cpu().numpy()}


def _compute_secret_gen_metrics(losses, loss_funcs, secret_gen_output, mels):
    # TODO: Need to fix this!!!
    all_fake_mel = torch.cat((secret_gen_output['faked_mel'], secret_gen_output['alt_faked_mel']), dim=1)
    male_mels = torch.index_select(all_fake_mel, 1, secret_gen_output['fake_secret'])
    female_mels = torch.index_select(all_fake_mel, 1, 1 - secret_gen_output['fake_secret'])

    all_fake_scores = torch.cat((secret_gen_output['fake_secret_score'], secret_gen_output['alt_fake_secret_score']), dim=1)
    male_scores = torch.index_select(all_fake_scores, 1, secret_gen_output['fake_secret'].type(torch.int))
    female_scores = torch.index_select(all_fake_scores, 1, 1 - secret_gen_output['fake_secret'].type(torch.int))

    male_female_diff = loss_funcs['distortion'](male_mels, female_mels)
    male_distortion = loss_funcs['distortion'](male_mels, mels)
    female_distortion = loss_funcs['distortion'](female_mels, mels)
    male_adversarial = loss_funcs['adversarial'](male_scores, secret_gen_output['fake_secret'])
    female_adversarial = loss_funcs['adversarial'](female_scores, secret_gen_output['fake_secret'])
   
    return {'distortion_loss': losses['secret_gen']['distortion'].detach().cpu().numpy(),
            'adversarial_loss': losses['secret_gen']['adversarial'].detach().cpu().numpy(),
            'combined_loss': losses['secret_gen']['final'].detach().cpu().numpy(),

            'male_distortion': male_distortion.detach().cpu().numpy(),
            'female_distortion': female_distortion.detach().cpu().numpy(),
            'male_adversarial': male_adversarial.detach().cpu().numpy(),
            'female_adversarial': female_adversarial.detach().cpu().numpy(),
            # 'male_final': male_final.detach().cpu().numpy(),
            # 'female_final': female_final.detach().cpu().numpy(),
            'male_femal_diff': male_female_diff.detach().cpu().numpy(),
            }


def _compute_filter_disc_metrics(losses, filter_disc_output, secret):
    filtered_secret_preds_disc = torch.argmax(filter_disc_output['filtered_secret_score'], 1).cpu().numpy()
    filtered_secret_accuracy_disc = np.array(accuracy_score(secret.cpu().numpy(), filtered_secret_preds_disc))

    unfiltered_secret_preds_disc = torch.argmax(filter_disc_output['unfiltered_secret_score'], 1).cpu().numpy()
    unfiltered_secret_accuracy_disc = np.array(accuracy_score(secret.cpu().numpy(), unfiltered_secret_preds_disc))
    return {'filtered_loss': losses['filter_disc']['final'].detach().cpu().numpy(),
            'filtered_accuracy': filtered_secret_accuracy_disc,
            'unfiltered_loss': losses['filter_disc']['unfiltered_score_loss'].detach().cpu().numpy(),
            'unfiltered_accuracy': unfiltered_secret_accuracy_disc}


def _compute_secret_disc_metrics(losses, secret_disc_output, secret_gen_output, secret):
    fake_secret_preds_disc = torch.argmax(secret_disc_output['fake_secret_score'], 1).cpu().numpy()
    fake_secret_label_accuracy_disc = np.array(accuracy_score(secret_disc_output['fake_secret'].cpu().numpy(),
                                                              fake_secret_preds_disc))
    real_secret_preds_disc = torch.argmax(secret_disc_output['real_secret_score'], 1).cpu().numpy()
    real_secret_label_accuracy_disc = np.array(accuracy_score(secret.cpu().numpy(), real_secret_preds_disc))
    generated_secret_accuracy_disc = np.array(accuracy_score(secret_gen_output['fake_secret'].cpu().numpy(),
                                                             fake_secret_preds_disc))

    return {'real_loss': losses['secret_disc']['real'].detach().cpu().numpy(),
            'fake_loss': losses['secret_disc']['fake'].detach().cpu().numpy(),
            'combined_loss': losses['secret_disc']['final'].detach().cpu().numpy(),
            'fake_accuracy': fake_secret_label_accuracy_disc,
            'real_accuracy': real_secret_label_accuracy_disc,
            'generated_accuracy': generated_secret_accuracy_disc}


def compute_metrics(mels, secret, label, filter_gen_output, filter_disc_output, secret_gen_output,
                    secret_disc_output, losses, loss_funcs):
    filter_gen_metrics = _compute_filter_gen_metrics(losses)
    secret_gen_metrics = _compute_secret_gen_metrics(losses, loss_funcs, secret_gen_output, mels)
    filter_disc_metrics = _compute_filter_disc_metrics(losses, filter_disc_output, secret)
    secret_disc_metrics = _compute_secret_disc_metrics(losses, secret_disc_output, secret_gen_output, secret)

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
