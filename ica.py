from sklearn.decomposition import FastICA


def get_ica_components(signal: [], no_components: int, random_state=69):
    ica = FastICA(no_components, tol=1e-3, max_iter=1000, random_state=random_state)
    ica_signals = ica.fit_transform(signal)  # Reconstruct signals

    return ica_signals
