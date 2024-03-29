from experiment_runs import AVAILABLE_RUNS
from datasets import AvailableDatasets
from training import init_training
import argparse
import configs
import torch
import tqdm


class WorkerConfig:
    def __init__(self, device):
        self.device = device
        self.available = True

    def assignJob(self, proc):
        self.currentJob = proc


AVAILABLE_DATASETS = {
    'audiomnist': AvailableDatasets.AudioMNIST,
    'cremad': AvailableDatasets.CremaD
}

# available_settings = {
#     'debug': configs.create_debug_config(),
#     'github_default': configs.create_github_default_config(),
#     'github_lower_lr': configs.create_github_lower_lr_config(),
#     'deterministic': configs.create_deterministic_config()
# }


def parallel_experiment_wrapper(queue, experiment_setting, dataset, device):
    experiment_setting.training.device = device
    print("--------------------------------------------------------------------")
    print(f" Start training on '{dataset.name}' dataset with '{str(experiment_setting)}' setting on {device}")
    print("--------------------------------------------------------------------")

    init_training(dataset=dataset, experiment_setup=experiment_setting, device=device)

    print("Finishing experiment on device {}".format(device))
    queue.put(device)


def parallel_whisper_experiment_wrapper(queue, experiment_setting, dataset, device):
    from whisper_gender_classification import main
    print("--------------------------------------------------------------------")
    print(f" Start whisper gender classification training on '{dataset.name}' dataset with '{str(experiment_setting)}' setting on {device}")
    print("--------------------------------------------------------------------")
    main(dataset=dataset, settings=experiment_setting, device=device)

    print("Finishing experiment on device {}".format(device))
    queue.put(device)


def parallel_experiment_queue(experiment, queue, dataset, workers):
    print('Experiment queue has length', len(queue))

    import torch.multiprocessing as mp
    mp.set_start_method('spawn')

    messageQueue = mp.Queue()

    for worker in workers:
        messageQueue.put(worker.device)

    processes = {}
    for experiment_setting in tqdm.tqdm(queue):
        device = messageQueue.get()
        if experiment == 'whisper':
            p = mp.Process(target=parallel_whisper_experiment_wrapper, args=(messageQueue, experiment_setting, dataset, device))
        else:
            p = mp.Process(target=parallel_experiment_wrapper, args=(messageQueue, experiment_setting, dataset, device))
        p.start()
        processes[device] = p

    for p in processes.values():
        p.join()


def verify_args(args):
    assert args['dataset'] in AVAILABLE_DATASETS
    assert args['experiment'] in AVAILABLE_RUNS
    from typing import List
    assert isinstance(args['gpus'], int) or \
           isinstance(args['gpus'], List) and all(isinstance(gpu, int) for gpu in args['gpus'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="name of dataset", default='audiomnist')
    parser.add_argument("-e", "--experiment", help="which pre-defined experiment queue to run", default='debug')
    parser.add_argument('-g', "--gpus", nargs="+", type=int, help="specify which gpus to use. Enter '-1' to use cpu."
                                                                  "Will also use cpu if no gpu is available")
    args = parser.parse_args()
    args = vars(args)

    args = {'dataset': 'cremad', 'experiment': 'whisper', 'gpus': [-1]}
    verify_args(args)

    if not torch.cuda.is_available() or args['gpus'] == [-1]:
        print("Using CPU worker:")
        workers = [WorkerConfig('cpu')]
    elif (len(args['gpus']) <= 0):
        raise Exception("No devices specified")
    else:
        print("Using GPU workers:", args['gpus'])
        workers = [WorkerConfig('cuda:{}'.format(i)) for i in args['gpus']]

    parallel_experiment_queue(args['experiment'], AVAILABLE_RUNS[args['experiment']], AVAILABLE_DATASETS[args['dataset']], workers)
