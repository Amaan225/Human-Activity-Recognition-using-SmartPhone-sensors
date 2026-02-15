import pandas as pd

# Path to training data
X_train_path = "data/UCI HAR Dataset/train/X_train.txt"
y_train_path = "data/UCI HAR Dataset/train/y_train.txt"

# Load data
X_train = pd.read_csv(X_train_path, sep=r"\s+", header=None)

y_train = pd.read_csv(y_train_path, header=None)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("\nFirst 5 rows of X_train:")
print(X_train.head())

print("\nFirst 5 labels:")
print(y_train.head())
