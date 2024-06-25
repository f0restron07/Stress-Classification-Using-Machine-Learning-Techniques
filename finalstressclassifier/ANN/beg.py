import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv(r"C:\Users\ghotn\Downloads\beg_data.csv")

feature_cols = ['Age', 'SB', 'HB', 'PSS']
x = df[feature_cols]
y = df.Class

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

mlp = MLPClassifier(solver='sgd', hidden_layer_sizes=(3,), random_state=0)

mlp.fit(x_train, y_train)

y_pred = mlp.predict(x_test)

accuracy_percentage = mlp.score(x_test, y_test) * 100


print("weights between input and 1st layer:")
print(mlp.coefs_[0])

print("Confusion Matrix: \n", metrics.confusion_matrix(y_test, y_pred))

print("Accuracy: {:.2f}%".format(metrics.accuracy_score(y_test, y_pred) * 100))

precision_percentage = metrics.precision_score(y_test, y_pred, average='macro') * 100
print("Precision: {:.2f}%".format(precision_percentage))
