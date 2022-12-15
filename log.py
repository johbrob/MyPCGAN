import wandb
import time
import numpy as np


def init(config_data, project='MyPCGAN', entity='johbrob', run_name='tmp-{}', disabled=False,
         add_timestamp_to_name=False):
    timestamp = time.time()
    config_data['model_ID'] = timestamp

    run_name = run_name.format(timestamp) if add_timestamp_to_name else run_name.format('')
    mode = 'disabled' if disabled else None

    wandb.init(name=run_name, config=config_data, project=project, entity=entity, mode=mode)


def _log_values(data, step, commit=True):
    wandb.log(data, step=step, commit=commit)


def metrics(data, step, prefix=None, suffix=None, aggregation=np.mean, selectedKeys=None, commit=False):
    prefix = prefix + '-' if prefix else ''
    suffix = '-' + suffix if suffix else ''

    targetKeys = selectedKeys if selectedKeys is not None else data.keys()
    data = {"{}{}{}".format(prefix, k, suffix): aggregation(data[k]) for k in targetKeys}
    # data['Epoch'] = epoch
    # data['Step'] = step
    _log_values(data, step, commit=commit)
