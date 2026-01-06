import numpy as np

def add_noise(X, noise_level=0.01):
    noisy = X.copy()
    noise = np.random.normal(0, noise_level, noisy.shape)
    return noisy + noise
