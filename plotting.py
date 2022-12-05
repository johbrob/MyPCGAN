import torch
import matplotlib.pyplot as plt
import librosa.display


def comparison_plot_filtergen(file_path, orig_spec, male_spec, female_spec, orig_title, male_title, female_title):
    orig_spec_np = torch.squeeze(orig_spec).cpu().numpy()
    male_spec_np = torch.squeeze(male_spec).cpu().numpy()
    female_spec_np = torch.squeeze(female_spec).cpu().numpy()
    fig = plt.figure(figsize=(30, 8))
    ax1 = fig.add_subplot(131)
    p1 = librosa.display.specshow(orig_spec_np, x_axis='time', y_axis='mel', sr=8000, fmax=4000, hop_length=256)
    plt.title(orig_title, fontsize=18)
    ax2 = fig.add_subplot(132)
    p2 = librosa.display.specshow(male_spec_np, x_axis='time', y_axis='mel', sr=8000, fmax=4000, hop_length=256)
    plt.title(male_title, fontsize=18)
    ax3 = fig.add_subplot(133)
    p3 = librosa.display.specshow(female_spec_np, x_axis='time', y_axis='mel', sr=8000, fmax=4000, hop_length=256)
    plt.title(female_title, fontsize=18)
    fig.savefig(file_path)
    plt.close(fig)


def comparison_plot_pcgan(file_path, orig_spec, filtered_spec, male_spec, female_spec, orig_title, filtered_title,
                          male_title, female_title):
    orig_spec_np = torch.squeeze(orig_spec).cpu().numpy()
    filtered_spec_np = torch.squeeze(filtered_spec).cpu().numpy()
    male_spec_np = torch.squeeze(male_spec).cpu().numpy()
    female_spec_np = torch.squeeze(female_spec).cpu().numpy()
    fig = plt.figure(figsize=(24, 24))  # This has to be changed!!
    ax1 = fig.add_subplot(221)
    p1 = librosa.display.specshow(orig_spec_np, x_axis='time', y_axis='mel', sr=8000, fmax=4000, hop_length=256,
                                  cmap='magma')
    plt.title(orig_title, fontsize=20)
    ax2 = fig.add_subplot(222)
    p2 = librosa.display.specshow(filtered_spec_np, x_axis='time', y_axis='mel', sr=8000, fmax=4000, hop_length=256,
                                  cmap='magma')
    plt.title(filtered_title, fontsize=20)
    ax3 = fig.add_subplot(223)
    p3 = librosa.display.specshow(male_spec_np, x_axis='time', y_axis='mel', sr=8000, fmax=4000, hop_length=256,
                                  cmap='magma')
    plt.title(male_title, fontsize=20)
    ax4 = fig.add_subplot(224)
    p4 = librosa.display.specshow(female_spec_np, x_axis='time', y_axis='mel', sr=8000, fmax=4000, hop_length=256,
                                  cmap='magma')
    plt.title(female_title, fontsize=20)
    fig.savefig(file_path)
    plt.close(fig)
