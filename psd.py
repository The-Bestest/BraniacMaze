import numpy as np
from scipy import signal


def get_psd_features(icas, sampling_rate):
    features = []

    for ica in icas.T:
        freq, psd = signal.periodogram(ica, fs=sampling_rate, nfft=512)

        # Only PSD at frequency between 2 and 45 is relevant
        lower_index = np.ix_(freq > 2)[0][0]  # first index that has frequency > 2
        upper_index = np.ix_(freq < 45)[0][-1]  # last index that has frequency < 45
        features.append(psd[lower_index:upper_index])

    return np.array(features)
