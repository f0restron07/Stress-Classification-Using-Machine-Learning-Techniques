import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Names of datasets for labeling in the graph
dataset_names = ['beg_data', 'after_task', 'after_med']

# Paths of the datasets
dataset_paths = [
    r'C:\Users\ghotn\Downloads\beg_data.csv',
    r'C:\Users\ghotn\Downloads\after_task.csv',
    r'C:\Users\ghotn\Downloads\after_med.csv'
]

# Initialize lists to store metrics for each dataset
metrics_data = {
    'accuracy': [],
    'precision': []
}

for dataset_path in dataset_paths:
    # Load dataset
    df = pd.read_csv(dataset_path)

    # Standardization
    features = ['Age', 'SB', 'DB', 'HB', 'PSS']
    X = df[features].values
    y = df['Class'].values
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # PCA Transformation
    cov_matrix = np.cov(X_std.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    eigen_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:,i]) for i in range(len(eigenvalues))]
    eigen_pairs.sort(key=lambda x: x[0], reverse=True)
    feature_vector = np.hstack([eigen_pairs[i][1].reshape(X_std.shape[1], 1) for i in range(len(eigenvalues))])
    X_pca = X_std.dot(feature_vector)

    # Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

    # TensorFlow Neural Network Model
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=100, validation_split=0.1, verbose=0)

    # Predictions and evaluation
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)

    # Append results to the lists
    metrics_data['accuracy'].append(accuracy * 100)
    metrics_data['precision'].append(precision * 100)

# Plotting the results
fig, ax = plt.subplots(figsize=(10, 6))

colors = ['b', 'g']
metrics_list = ['accuracy', 'precision']
for metric, color in zip(metrics_list, colors):
    ax.plot(dataset_names, metrics_data[metric], 'o-', color=color, label=f'{metric.capitalize()} (%)')

ax.set_xlabel('Dataset')
ax.set_ylabel('Percentage')
ax.set_title('Comparison of Neural Network Performance Across Datasets')
ax.legend()

plt.show()
