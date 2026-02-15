import numpy as np
from features.filters import highpass_filter

def extract_fusion_features(window, fs=100):
    """
    window shape: (N, 6)
    columns: ax, ay, az, gx, gy, gz
    """

    arr = np.array(window)

    ax, ay, az = arr[:, 0], arr[:, 1], arr[:, 2]
    gx, gy, gz = arr[:, 3], arr[:, 4], arr[:, 5]

    # 🔴 Remove gravity from accelerometer
    ax = highpass_filter(ax, fs)
    ay = highpass_filter(ay, fs)
    az = highpass_filter(az, fs)

    # Magnitudes
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)

    # Jerk (temporal dynamics)
    acc_jerk = np.diff(acc_mag)
    gyro_jerk = np.diff(gyro_mag)

    feats = []

    # Time-domain stats
    def stats(x):
        return [np.mean(x), np.std(x), np.min(x), np.max(x)]

    for axis in (ax, ay, az):
        feats.extend(stats(axis))

    for axis in (gx, gy, gz):
        feats.extend(stats(axis))

    feats.extend(stats(acc_mag))
    feats.extend(stats(gyro_mag))
    feats.extend(stats(acc_jerk))
    feats.extend(stats(gyro_jerk))

    # 🔴 Frequency-domain energy (KEY for walking vs running)
    acc_fft = np.abs(np.fft.rfft(acc_mag))
    gyro_fft = np.abs(np.fft.rfft(gyro_mag))

    feats.extend([
        np.mean(acc_fft),
        np.std(acc_fft),
        np.max(acc_fft),
        np.mean(gyro_fft),
        np.std(gyro_fft),
        np.max(gyro_fft),
    ])

    return np.array(feats, dtype=np.float32).reshape(1, -1)
