import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score

import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\Users\ghotn\Downloads\after_task.csv") 


# Extract features and target variable
X = df[['Age', 'SB', 'DB', 'HB', 'PSS']].values
y = df['Class'].values


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_reshaped = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))


X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)


model = Sequential()
model.add(SimpleRNN(64, input_shape=(1, X_train.shape[2]), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=100, validation_split=0.1, verbose=1)


loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')



y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype("int32")
cm = confusion_matrix(y_test, y_pred_classes)

accuracy = accuracy_score(y_test, y_pred_classes)
print(f'Accuracy: {accuracy:.2%}')

precision = precision_score(y_test, y_pred_classes)
print(f'Precision: {precision:.2%}')


fig, ax = plt.subplots(figsize=(8, 8))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


weights_rnn = model.layers[0].get_weights()[0]
print(weights_rnn)