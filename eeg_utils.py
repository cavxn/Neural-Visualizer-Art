import numpy as np
from scipy.fft import rfft, rfftfreq

# EEG Band Ranges (in Hz)
bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 50),
}

# Get dominant EEG band using FFT
def get_dominant_band(eeg_row, sampling_rate=128):
    eeg_row = np.array(eeg_row)
    yf = rfft(eeg_row)
    xf = rfftfreq(len(eeg_row), 1 / sampling_rate)
    power = np.abs(yf)**2

    band_powers = {}
    for band, (low, high) in bands.items():
        idx = np.where((xf >= low) & (xf <= high))[0]
        band_powers[band] = np.sum(power[idx])

    return max(band_powers, key=band_powers.get)
