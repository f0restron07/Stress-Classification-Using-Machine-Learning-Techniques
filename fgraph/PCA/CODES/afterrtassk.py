import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, log_loss
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import Precision

df = pd.read_csv(r'C:\Users\ghotn\Downloads\after_task.csv')

features = ['Age', 'SB', 'DB', 'HB', 'PSS']  
X = df[features].values
y = df['Class'].values
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
cov_matrix = np.cov(X_std.T)


eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)


eigen_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:,i]) for i in range(len(eigenvalues))]
eigen_pairs.sort(key=lambda x: x[0], reverse=True)

feature_vector = np.hstack([eigen_pairs[i][1].reshape(X_std.shape[1], 1) for i in range(len(eigenvalues))])


X_pca = X_std.dot(feature_vector)


X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)


model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision()])


history = model.fit(X_train, y_train, epochs=100, validation_split=0.1, verbose=0)

train_loss_change = np.diff(history.history['loss'])
val_loss_change = np.diff(history.history['val_loss'])

accuracy_changes = history.history['accuracy']
precision_changes = history.history['precision']


plt.figure(figsize=(10, 5))
plt.plot(train_loss_change, label='Train Loss Change', color='blue')
plt.plot(val_loss_change, label='Validation Loss Change', color='orange')
plt.title('Rate of Change of Loss')
plt.ylabel('Loss Change')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(accuracy_changes, label='Accuracy', color='blue')
plt.plot(precision_changes, label='Precision', color='orange')
plt.title('Accuracy and Precision Over Epochs')
plt.ylabel('Metrics')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()
