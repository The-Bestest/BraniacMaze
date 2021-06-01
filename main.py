import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import loadmat
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from bandpass import butter_bandpass_filter
from ica import get_ica_components
from psd import get_psd_features

sampling_rate = 1000  # signal is at 1000Hz
lowcut = 1.0
highcut = 45.0

seed = 69
results_filepath = "results.txt"


def get_sliding_windows(sequence, window_size, step=1):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable."""

    # Verify the inputs
    try:
        it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(window_size) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > window_size:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if window_size > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence length.")

    # Pre-compute number of chunks to emit
    numOfChunks = int((len(sequence) - window_size) / step) + 1

    # Do the work
    for i in range(0, numOfChunks * step, step):
        yield (i, sequence[i:i + window_size])


def get_channels(data, channel_ids=None, from_sample=None, to_sample=None):
    if not channel_ids:
        channel_ids = list(range(data.shape[0]))

    if not from_sample:
        from_sample = 0

    if not to_sample:
        to_sample = data.shape[0]

    return data[from_sample:to_sample].T[channel_ids]


def get_markers(matlab_data):
    """
    :return: dense array of markers
    """
    time = matlab_data['mrk'][0][0][0][0]
    labels = matlab_data['mrk'][0][0][1][0]

    return np.array([time, labels]).T


def plot_time_domain(channels_signal: []):
    """
    Plots signals in time domain
    :param channels_signal: signal of all channels to plot
    :return: None
    """
    for y in channels_signal:
        x = list(range(len(y)))
        plt.plot(x, y)


def plot_frequency_domain(signals: [], max_frequency: int = 100):
    """
    Plots signals in frequency domain
    :param signals: signal of all channels to plot
    :param max_frequency: Cutoff in graph beyond which we hide data
    :return: None
    """
    plt.title('PSD: power spectral density')
    plt.tight_layout(pad=1.8)

    for idx, signal_data in enumerate(signals):
        freq, psd = signal.periodogram(signal_data, fs=sampling_rate, nfft=max_frequency)

        plt.subplot(len(signals), 1, idx + 1)
        plt.plot(freq, psd, color='red')
        plt.xlim(0, max_frequency)


def plot_signal_comparison(signals: [], names: []):
    for ii, (signal_data, name) in enumerate(zip(signals, names), 1):
        plt.subplot(len(signals), 1, ii)
        plt.title(name)
        plt.plot(signal_data)


def plot_features(features: []):
    plt.subplots_adjust(left=0.12,
                        bottom=0.08,
                        right=0.95,
                        top=0.95,
                        wspace=0.2,
                        hspace=0.35)

    for idx, signal_data in enumerate(features):
        plt.subplot(len(features), 1, idx + 1)
        x = list(range(len(signal_data)))
        plt.xlim(-0.2, len(x))
        plt.scatter(x, signal_data)


def preprocess(windows, markers=None):
    all_features = []
    all_labels = []
    for idx, (sample_no, window) in enumerate(windows):
        if markers is not None:
            # Find all markers that happen inside the first half of the current window
            # The reason why we don't care if the label is in the second half of the window is,
            # that the subject could not have reacted to the cue so fast
            mask = np.logical_and(markers[:, 0] > sample_no, markers[:, 0] < sample_no + len(window) / 2)
            selected = markers[mask]

            if selected.size > 0:
                # We only care about the first label that happens in the window
                # as only that one is probably relevant
                all_labels.append(selected[0][1])
            else:
                # No markers means this window is entirely rest
                all_labels.append(0)

        eeg_signal = np.array(window)
        filtered_channels = np.array(list(map(
            lambda channel: butter_bandpass_filter(channel, lowcut, highcut, sampling_rate, order=4),
            eeg_signal.T))).T

        icas = get_ica_components(filtered_channels, filtered_channels.shape[1], random_state=seed)
        features = get_psd_features(icas, sampling_rate)
        # We are flattening the feature set, since none of the off-the-shelf implementations of mutual information
        # scores and LSVM that we could get our hands on, deals with the higher-dimensional features.
        # Note that a higher-dimensional feature is different from high number of features.
        # This means that we are losing the analogy of '1 feature = 1 brain source', but this might not matter for
        # the accuracy at the end, as we cannot ensure that 1 feature = 1 brain source will always hold true anyway,
        # or at least that each feature is always a mix of the same brain sources, as the features are only numerically
        # estimated using ICA without any prior knowledge (like sensor locations).
        # Of course, this is just a guess and we have no way to test it.
        all_features.append(features.flatten())

    return all_features, all_labels


def load_data(filepath, has_labels):
    matlab = loadmat(filepath)
    data = np.array(matlab['cnt'])

    channels = get_channels(data, [1, 2, 5])
    markers = get_markers(matlab) if has_labels else None

    return channels, markers


def make_data(participant_filepath, window_size=2000, has_labels=True):
    channels_train, markers_train = load_data(participant_filepath, has_labels)

    windows_train = get_sliding_windows(channels_train.T, window_size, 500)

    return preprocess(windows_train, markers_train)


def run(filepath, idx):
    print("Preprocessing calibration data for {0}".format(filepath))
    X_train, Y_train = make_data(filepath, has_labels=True)

    param_grid = {
        'selectkbest__k': [1, 30],
        'linearsvc__C': [0.1, 1, 10, 100]
    }

    print("Learning from calibration data for {0}".format(filepath))
    pipeline = make_pipeline(
        StandardScaler(),
        SelectKBest(mutual_info_classif, k='all'),
        StandardScaler(),
        LinearSVC(max_iter=1000, tol=1e-3, random_state=seed, class_weight='balanced', multi_class='ovr')
    )

    hyperparams_optimiser = GridSearchCV(pipeline, param_grid=param_grid, cv=StratifiedKFold(n_splits=5),
                                         refit=True).fit(X_train, Y_train)
    print("Best score: {0}".format(hyperparams_optimiser.best_score_))
    print("Best params: {0}".format(hyperparams_optimiser.best_params_))

    with open(results_filepath, 'a') as f:
        f.write(filepath + '\n')
        f.write("Best score: {0}\n".format(hyperparams_optimiser.best_score_))
        f.write("Best params: {0}\n".format(hyperparams_optimiser.best_params_))
        f.write('\n')

    prediction_test = predict(hyperparams_optimiser, path[0])
    plot_eval(prediction_test, idx, Y_train)
    np.savetxt("prediction_test.txt", prediction_test, delimiter=' ', newline='\n')

    return hyperparams_optimiser


def predict(pipeline, filepath):
    print("Preprocessing eval data for {0}".format(filepath))
    X_eval, _ = make_data(filepath, has_labels=False)

    print("Predicting eval data for {0}".format(filepath))
    return pipeline.predict(X_eval)


def plot_eval(data, subject_no, ground_truth=None):
    session_type = "calibration" if ground_truth else "evaluation"
    plt.scatter(range(len(data)), data)
    if ground_truth:
        correct = []
        for p, g in zip(data, ground_truth):
            if p == g:
                correct.append(None)
            else:
                correct.append(g)

        plt.scatter(range(len(ground_truth)), correct, c="orange")
    plt.ylabel("Event type")
    plt.xlabel("Sample")
    plt.title("Predicted motor imagery event types\nduring {} session for subject {}".format(session_type, subject_no))
    plt.show()


if __name__ == "__main__":
    results = []

    files = [('calibration/{0}'.format(file[0]), 'evaluation/{0}'.format(file[1])) for file in
             zip(sorted(os.listdir('calibration')), sorted(os.listdir('evaluation')))]

    if os.path.exists(results_filepath):
        os.remove(results_filepath)
    for idx, path in enumerate(files[:]):
        clf = run(path[0], idx + 1)
        results.append(clf.best_score_)

        prediction_eval = predict(clf, path[1])
        plot_eval(prediction_eval, idx + 1)
        np.savetxt("prediction_eval.txt", prediction_eval, delimiter=' ', newline='\n')

    plt.bar(list(range(len(results))), results)
    plt.xlabel("Participant")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy score")
    plt.title("Accuracy scores for the best model found \nacross calibration sessions of different participants")
    plt.show()
