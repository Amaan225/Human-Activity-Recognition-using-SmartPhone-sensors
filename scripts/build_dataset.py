import os
import numpy as np
import pandas as pd
from features.fusion_features import extract_fusion_features

import os

os.makedirs("processed_data", exist_ok=True)


DATASET_DIR = "dataset"
WINDOW = 256
STEP = 128

LABELS = {
    "standing": 0,
    "sitting": 1,
    "laying": 2,
    "walking": 3,
    "running": 4
}

X, y = [], []

for activity in LABELS:
    activity_dir = os.path.join(DATASET_DIR, activity)

    for session in os.listdir(activity_dir):
        session_dir = os.path.join(activity_dir, session)

        acc = pd.read_csv(os.path.join(session_dir, "Accelerometer.csv"))
        gyr = pd.read_csv(os.path.join(session_dir, "Gyroscope.csv"))

        acc_data = acc.iloc[:, 1:4].values
        gyr_data = gyr.iloc[:, 1:4].values

        min_len = min(len(acc_data), len(gyr_data))

        acc_data = acc_data[:min_len]
        gyr_data = gyr_data[:min_len]

        data = np.hstack([acc_data, gyr_data])


        for start in range(0, len(data) - WINDOW, STEP):
            window = data[start:start + WINDOW]
            feats = extract_fusion_features(window)
            X.append(feats[0])
            y.append(LABELS[activity])

X = np.array(X)
y = np.array(y)

np.save("processed_data/X.npy", X)
np.save("processed_data/y.npy", y)

print("Dataset built:", X.shape, y.shape)
