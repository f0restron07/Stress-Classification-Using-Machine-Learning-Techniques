import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, log_loss
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


df = pd.read_csv(r'C:\Users\ghotn\Downloads\beg_data.csv')

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


X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}


model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=100, validation_split=0.1, class_weight=class_weights_dict, verbose=0)


y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)


loss = log_loss(y_test, y_pred_proba, labels=[0, 1])

accuracy = accuracy_score(y_test, y_pred)
# Handle case where y_pred is all zeros which would lead to undefined precision
precision = precision_score(y_test, y_pred, zero_division=1)

print(f"\nAccuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Log Loss: {loss:.4f}")
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nCovariance matrix:\n", cov_matrix)
print("\nEigenvalues:\n", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)
print("\nFeature Vector (PCA Components):\n", feature_vector)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()