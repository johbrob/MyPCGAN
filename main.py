from DataManagers import AudioMNIST_manager


def main():
    num_genders = 2
    num_digits = 10

    data_file = '/home/johbro/Code/FederatedLearning/AudioMNIST/'
    annotation_file = '/home/johbro/Code/FederatedLearning/AudioMNIST/data/audioMNIST_meta.txt'

    file_idx, an_gender_idx, \
    an_digit_idx, an_speaker_id = AudioMNIST_manager.build_annotation_index(data_file, annotation_file)

    print("create datasets")


if __name__ == '__main__':
    main()