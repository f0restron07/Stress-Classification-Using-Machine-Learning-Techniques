import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Paths to the datasets
datasets = {
    'beg_data': r"C:\Users\ghotn\Downloads\beg_data.csv",
    'after_task': r"C:\Users\ghotn\Downloads\after_task.csv",
    'after_med': r"C:\Users\ghotn\Downloads\after_med.csv"
}

# Lists to store results
accuracies = []
precisions = []

# Processing each dataset
for name, path in datasets.items():
    # Load dataset
    df = pd.read_csv(path)

    # Preparing the Data
    X = df[['Age', 'SB', 'DB', 'HB', 'PSS']].values
    y = df['Class'].values

    # Data Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_reshaped = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

    # Build RNN model
    model = Sequential()
    model.add(SimpleRNN(64, input_shape=(1, X_train.shape[2]), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=100, validation_split=0.1, verbose=0)

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype("int32")

    # Calculate accuracy and precision
    acc = accuracy_score(y_test, y_pred_classes) * 100
    prec = precision_score(y_test, y_pred_classes) * 100

    # Store results
    accuracies.append(acc)
    precisions.append(prec)

# Plotting the results
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(list(datasets.keys()), accuracies, 'o-', label='Accuracy (%)')
ax.plot(list(datasets.keys()), precisions, 's--', label='Precision (%)')
ax.set_xlabel('Dataset')
ax.set_ylabel('Performance (%)')
ax.set_title('RNN Model Performance Across Datasets')
ax.legend()
plt.show()
