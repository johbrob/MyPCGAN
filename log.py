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


def data(data):
    pass


def _log_values(data, commit=True):
    wandb.log(data, commit=commit)


def metrics(data, prefix=None, suffix=None, aggregation=np.mean, selectedKeys=None, commit=False):
    prefix = prefix + '-' if prefix else ''
    suffix = '-' + suffix if suffix else ''

    targetKeys = selectedKeys if selectedKeys is not None else data.keys()
    _log_values({"{}{}{}".format(prefix, k, suffix): aggregation(data[k]) for k in targetKeys}, commit=commit)
