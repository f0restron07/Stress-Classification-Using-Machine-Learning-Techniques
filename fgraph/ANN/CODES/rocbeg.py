import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import log_loss, accuracy_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\ghotn\Downloads\beg_data.csv")

feature_cols = ['Age', 'SB', 'HB', 'PSS']  # Ensured to be correct as per your latest update
X = df[feature_cols]
y = df['Class']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=1)
for train_index, val_index in sss.split(X_train, y_train):
    X_train_part, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_part, y_val = y_train.iloc[train_index], y_train.iloc[val_index]


mlp = MLPClassifier(solver='sgd', hidden_layer_sizes=(3,), random_state=0, max_iter=1, warm_start=True)

accuracies = []
precisions = []
train_losses = []
val_losses = []


# Determine all possible classes from the entire dataset
all_classes = np.unique(y)

for i in range(100):  # Number of epochs
    mlp.partial_fit(X_train_part, y_train_part, classes=all_classes)
    

    y_train_pred_proba = mlp.predict_proba(X_train_part)
    y_val_pred_proba = mlp.predict_proba(X_val)
    y_pred = mlp.predict(X_test)
    

    train_loss = log_loss(y_train_part, y_train_pred_proba, labels=all_classes)
    val_loss = log_loss(y_val, y_val_pred_proba, labels=all_classes)
    

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    accuracies.append(accuracy)
    precisions.append(precision)



accuracy_changes = np.diff(accuracies)
precision_changes = np.diff(precisions)


train_loss_changes = np.diff(train_losses)
val_loss_changes = np.diff(val_losses)

plt.figure(figsize=(12, 6))
plt.plot(accuracy_changes, label='Accuracy Change', color='blue')
plt.plot(precision_changes, label='Precision Change', color='orange')
plt.title('Rate of Change of Accuracy and Precision')
plt.xlabel('Epoch')
plt.ylabel('Metric Change')
plt.legend()
plt.grid(True)
plt.show()

# Plot training and validation loss changes
plt.figure(figsize=(12, 6))
plt.plot(train_loss_changes, label='Training Loss Change', color='blue')
plt.plot(val_loss_changes, label='Validation Loss Change', color='orange')
plt.title('Rate of Change of Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Change')
plt.legend()
plt.grid(True)
plt.show()


