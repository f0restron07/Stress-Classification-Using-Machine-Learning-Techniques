import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Names of the datasets for labeling in the graph in the desired order
dataset_names = ['beg_data', 'after_task', 'after_med']

# Initialize lists to store metrics for each dataset
metrics_data = {
    'accuracy': [],
    'precision': []
}

# List of dataset paths in the new order
dataset_paths = [
    r"C:\Users\ghotn\Downloads\beg_data.csv",
    r"C:\Users\ghotn\Downloads\after_task.csv",
    r"C:\Users\ghotn\Downloads\after_med.csv"
]

for dataset_path in dataset_paths:
    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Define features and target variable depending on the dataset
    if 'd after m' in df.columns:
        feature_cols = ['Age', 's after m', 'd after m', 'hr after m', 'PSS']
    else:
        feature_cols = ['Age', 'SB', 'HB', 'PSS']
        
    x = df[feature_cols]
    y = df['Class']

    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # Initialize the MLPClassifier
    mlp = MLPClassifier(solver='sgd', hidden_layer_sizes=(3,), random_state=0)

    # Fit the model on the training data
    mlp.fit(x_train, y_train)

    # Evaluate the model
    accuracy = mlp.score(x_test, y_test) * 100
    precision = metrics.precision_score(y_test, mlp.predict(x_test), average='macro') * 100

    # Append results to the lists
    metrics_data['accuracy'].append(accuracy)
    metrics_data['precision'].append(precision)

# Plotting the results
fig, ax = plt.subplots(figsize=(10, 6))

# Different markers and styles for clearer differentiation
styles = ['o-', 's--', 'd:']

for metric, values in zip(metrics_data.keys(), metrics_data.values()):
    ax.plot(dataset_names, values, styles.pop(0), label=f'{metric.capitalize()} (%)')

ax.set_xlabel('Dataset')
ax.set_ylabel('Percentage')
ax.set_title('Comparison of MLP Classifier Performance Across Datasets')
ax.legend()
ax.grid(True)  # Adding grid for better readability

plt.show()
