import pandas as pd
import numpy as np

# Generate some random data for demonstration purposes
n_samples = 1000
n_features = 5

X = np.random.rand(n_samples, n_features)
y = np.random.randint(0, 2, n_samples)

# Split the data into training and testing sets
train_size = 0.8
split_index = int(n_samples * train_size)

X_train, y_train = X[:split_index], y[:split_index]
X_test, y_test = X[split_index:], y[split_index:]

# Save the data to a CSV file
data = np.hstack([X, y.reshape(-1, 1)])
df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(n_features)] + ["target"])

df.to_csv("test_data.csv", index=False)