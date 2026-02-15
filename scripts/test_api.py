import pandas as pd
import requests

# Load one real sample
X = pd.read_csv(
    "data/UCI HAR Dataset/train/X_train.txt",
    sep=r"\s+",
    header=None
)

sample = X.iloc[0].tolist()  # one row = 561 features

# Send request
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"features": sample}
)

print(response.json())
