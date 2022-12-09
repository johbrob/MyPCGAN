from DataManaging import AvailableDatasets
from training import init_training
import argparse
import configs
import torch

available_dataset = {
    'audiomnist': AvailableDatasets.AudioMNIST
}

available_settings = {
    'debug': configs.create_debug_config(),
    'github_default': configs.create_github_default_config(),
    'github_lower_lr': configs.create_github_lower_lr_config(),
}


def verify_args(args):
    assert args['dataset'] in available_dataset
    assert args['settings'] in available_settings
    assert isinstance(args['gpu'], int) or args['gpu'].isdigit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="name of dataset", default='audiomnist')
    parser.add_argument("-s", "--settings", help="which pre-defined setting to use", default='debug')
    parser.add_argument("-g", "--gpu", help="specify which gpu to use. Enter 'no' to use cpu. "
                                            "Will also use cpu if no gpu is available", default='0')
    args = parser.parse_args()
    args = vars(args)

    # args = {'dataset': 'audiomnist', 'settings': 'debug', 'gpu': 0}
    verify_args(args)

    if not torch.cuda.is_available() or args['gpu'] == 'no':
        device = torch.device('cpu')
    else:
        device = torch.device(int(args['gpu']))

    print("----------------------------------------------------------------")
    print(f" Start '{args['dataset']}' training with '{args['settings']}' setting on {device}")
    print("----------------------------------------------------------------")

    init_training(dataset_name=available_dataset[args['dataset']],
                  experiment_settings=available_settings[args['settings']], device=device)
