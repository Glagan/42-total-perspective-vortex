from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import mne
from mne import Epochs, pick_types, annotations_from_events, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP, SPoC
from mne.viz import plot_events, plot_montage
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs, corrmap, Xdawn

mne.set_log_level("CRITICAL")

tmin, tmax = -1.0, 4.0
subject = 1
runs = [3, 7, 11]
event_id = dict(rest=0, left=1, right=2)

# Open raw
raw_fnames = [f"dataset/S{subject:03d}/S{subject:03d}R{run:02d}.edf" for run in runs]
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
events, _ = events_from_annotations(raw, event_id=dict(T0=0, T1=1, T2=2))
eegbci.standardize(raw)  # set channel names
montage = make_standard_montage("standard_1020")
# montage.plot()
raw.set_montage(montage)

# Before filter
raw.plot_psd()
raw.plot_psd(average=True)
raw.plot(scalings=dict(eeg=200e-6))
plt.show()

# Apply band-pass filter
raw.notch_filter(60, method="iir")
raw.filter(7, 30.0, fir_design="firwin", skip_by_annotation="edge")
# print(raw.info)

# Select channels
# See https://arxiv.org/pdf/1312.2877.pdf -- page 3
channels = raw.info["ch_names"]
print(channels, len(channels))
good_channels = [
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
]
bad_channels = [x for x in channels if x not in good_channels]
raw.drop_channels(bad_channels)

# After filter
raw.plot_psd()
raw.plot_psd(average=True)
raw.plot(scalings=dict(eeg=200e-6))
plt.show()

# All events
fig = plot_events(events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp, event_id=event_id)
fig.subplots_adjust(right=0.7)  # make room for legend

# Read epochs
# Testing will be done with a running classifier
events, event_id = events_from_annotations(raw)
picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
labels = epochs.events[:, -1]
print("labels", labels)
