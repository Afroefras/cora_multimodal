from torch import Tensor
from numpy.random import shuffle as np_shuffle
from matplotlib.pyplot import subplots, show


def plot_sounds(
    sounds: Tensor,
    names: dict,
    labels: Tensor,
    n: int,
    spectrogram: bool,
    shuffle: bool,
) -> None:
    data_len = len(sounds)
    idxs = list(range(data_len))

    if shuffle:
        np_shuffle(idxs)

    to_show = idxs[:n]

    sounds = sounds.squeeze()

    n_rows = 2 if spectrogram else 1
    fig, axs = subplots(n_rows, n)

    fig.set_figwidth(40)
    fig.suptitle(f'{n} {"random" if shuffle else ""} records:')

    for i in range(n):
        idx = to_show[i]
        name = names[idx]
        sound = sounds[idx]
        label = labels[idx].item()

        label = "Good" if label > 0.5 else "Bad"

        axs[0, i].plot(sound)
        axs[0, i].set_title(name)
        axs[1, i].specgram(sound)
        axs[1, i].set_xlabel(label, fontsize=18)

    show()


# from wfdb import rdrecord, plot_wfdb

# record = rdrecord("data/physionet.org/files/ephnogram/1.0.0/WFDB/ECGPCG0004")
# plot_wfdb(record=record)


# import io
# import os
# import math
# import tarfile
# import multiprocessing

# import scipy
# import librosa
# import boto3
# from botocore import UNSIGNED
# from botocore.config import Config
# import requests
# import matplotlib
# import matplotlib.pyplot as plt
# import pandas as pd
# import time
# from IPython.display import Audio, display

# def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
#   waveform = waveform.numpy()

#   num_channels, num_frames = waveform.shape
#   time_axis = torch.arange(0, num_frames) / sample_rate

#   figure, axes = plt.subplots(num_channels, 1)
#   if num_channels == 1:
#     axes = [axes]
#   for c in range(num_channels):
#     axes[c].plot(time_axis, waveform[c], linewidth=1)
#     axes[c].grid(True)
#     if num_channels > 1:
#       axes[c].set_ylabel(f'Channel {c+1}')
#     if xlim:
#       axes[c].set_xlim(xlim)
#     if ylim:
#       axes[c].set_ylim(ylim)
#   figure.suptitle(title)
#   plt.show(block=False)

# def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
#   waveform = waveform.numpy()

#   num_channels, num_frames = waveform.shape
#   time_axis = torch.arange(0, num_frames) / sample_rate

#   figure, axes = plt.subplots(num_channels, 1)
#   if num_channels == 1:
#     axes = [axes]
#   for c in range(num_channels):
#     axes[c].specgram(waveform[c], Fs=sample_rate)
#     if num_channels > 1:
#       axes[c].set_ylabel(f'Channel {c+1}')
#     if xlim:
#       axes[c].set_xlim(xlim)
#   figure.suptitle(title)
#   plt.show(block=False)

# def play_audio(waveform, sample_rate):
#   waveform = waveform.numpy()

#   num_channels, num_frames = waveform.shape
#   if num_channels == 1:
#     display(Audio(waveform[0], rate=sample_rate))
#   elif num_channels == 2:
#     display(Audio((waveform[0], waveform[1]), rate=sample_rate))
#   else:
#     raise ValueError("Waveform with more than 2 channels are not supported.")

# def play_audio(waveform, sample_rate):
#   waveform = waveform.numpy()

#   num_channels, num_frames = waveform.shape
#   if num_channels == 1:
#     display(Audio(waveform[0], rate=sample_rate))
#   elif num_channels == 2:
#     display(Audio((waveform[0], waveform[1]), rate=sample_rate))
#   else:
#     raise ValueError("Waveform with more than 2 channels are not supported.")

# def inspect_file(path):
#   print("-" * 10)
#   print("Source:", path)
#   print("-" * 10)
#   print(f" - File size: {os.path.getsize(path)} bytes")
#   print(f" - {torchaudio.info(path)}")

# def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
#   fig, axs = plt.subplots(1, 1)
#   axs.set_title(title or 'Spectrogram (db)')
#   axs.set_ylabel(ylabel)
#   axs.set_xlabel('frame')
#   im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
#   if xmax:
#     axs.set_xlim((0, xmax))
#   fig.colorbar(im, ax=axs)
#   plt.show(block=False)

# def plot_mel_fbank(fbank, title=None):
#   fig, axs = plt.subplots(1, 1)
#   axs.set_title(title or 'Filter bank')
#   axs.imshow(fbank, aspect='auto')
#   axs.set_ylabel('frequency bin')
#   axs.set_xlabel('mel bin')
#   plt.show(block=False)

# def get_spectrogram(
#     n_fft = 400,
#     win_len = None,
#     hop_len = None,
#     power = 2.0,
# ):
#   waveform, _ = get_speech_sample()
#   spectrogram = T.Spectrogram(
#       n_fft=n_fft,
#       win_length=win_len,
#       hop_length=hop_len,
#       center=True,
#       pad_mode="reflect",
#       power=power,
#   )
#   return spectrogram(waveform)

# def plot_pitch(waveform, sample_rate, pitch):
#   figure, axis = plt.subplots(1, 1)
#   axis.set_title("Pitch Feature")
#   axis.grid(True)

#   end_time = waveform.shape[1] / sample_rate
#   time_axis = torch.linspace(0, end_time,  waveform.shape[1])
#   axis.plot(time_axis, waveform[0], linewidth=1, color='gray', alpha=0.3)

#   axis2 = axis.twinx()
#   time_axis = torch.linspace(0, end_time, pitch.shape[1])
#   ln2 = axis2.plot(
#       time_axis, pitch[0], linewidth=2, label='Pitch', color='green')

#   axis2.legend(loc=0)
#   plt.show(block=False)

# def plot_kaldi_pitch(waveform, sample_rate, pitch, nfcc):
#   figure, axis = plt.subplots(1, 1)
#   axis.set_title("Kaldi Pitch Feature")
#   axis.grid(True)

#   end_time = waveform.shape[1] / sample_rate
#   time_axis = torch.linspace(0, end_time,  waveform.shape[1])
#   axis.plot(time_axis, waveform[0], linewidth=1, color='gray', alpha=0.3)

#   time_axis = torch.linspace(0, end_time, pitch.shape[1])
#   ln1 = axis.plot(time_axis, pitch[0], linewidth=2, label='Pitch', color='green')
#   axis.set_ylim((-1.3, 1.3))

#   axis2 = axis.twinx()
#   time_axis = torch.linspace(0, end_time, nfcc.shape[1])
#   ln2 = axis2.plot(
#       time_axis, nfcc[0], linewidth=2, label='NFCC', color='blue', linestyle='--')

#   lns = ln1 + ln2
#   labels = [l.get_label() for l in lns]
#   axis.legend(lns, labels, loc=0)
#   plt.show(block=False)
