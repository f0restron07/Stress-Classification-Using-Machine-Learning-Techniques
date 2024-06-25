# Stress Classification Using Machine Learning Techniques

## Overview
This project aims to classify stress levels using machine learning techniques based on data collected from OpenBCI machines and OpenBCI GUI. The data was gathered from individuals before a task, after a task, and after meditation, encompassing various parameters such as Age, SB, DB, HB, PSS, and Class. The project leverages several machine learning algorithms, including ANN, RNN, and PCA (based on RNN), to achieve high precision and accuracy in stress classification.

## Table of Contents
- [Introduction](#introduction)
- [Data Collection](#data-collection)
- [Algorithms Implemented](#algorithms-implemented)
- [Performance Metrics](#performance-metrics)
- [Usage](#usage)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction
Stress is a prevalent issue that can significantly impact physical and mental health. This project aims to provide an efficient and accurate method for classifying stress levels using data collected through OpenBCI devices. By employing advanced machine learning techniques, we can analyze physiological data and predict stress levels with high precision.

## Data Collection
The data used in this project was collected using OpenBCI machines and the OpenBCI GUI. Data was recorded in three phases:
- **Before Task**
- **After Task**
- **After Meditation**

The parameters collected include:
- **Age**
- **SB (Skin Blood Flow)**
- **DB (Diastolic Blood Pressure)**
- **HB (Heart Beat Rate)**
- **PSS (Perceived Stress Scale)**
- **Class (Stress Level)**

## Algorithms Implemented
We implemented several machine learning algorithms to classify stress levels:
- **Artificial Neural Networks (ANN)**
- **Recurrent Neural Networks (RNN)**
- **Principal Component Analysis (PCA)** (based on RNN)

These algorithms were evaluated for their precision, accuracy, and overall performance.

## Performance Metrics
The performance of the implemented algorithms was measured using the following metrics:
- **Precision**
- **Accuracy**
- **Graphical Performance Analysis**

Graphs depicting the performance of each algorithm were generated to provide a visual comparison.

## Usage
This project can be used in various applications, including:
- **Health Monitoring**: Continuous monitoring of stress levels to prevent stress-related health issues.
- **Workplace Stress Management**: Identifying and managing stress in employees to improve productivity and well-being.
- **Mental Health Applications**: Assisting mental health professionals in diagnosing and treating stress-related conditions.
- **Research**: Providing a dataset and methodology for further research in stress classification and machine learning.

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/f0restron07/Stress-Classification-Using-Machine-Learning-Techniques.git
2. Navigate to the project directory:
   ```sh
   cd Stress-Classification-Using-Machine-Learning-Techniques
3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
4.Run the main script:
  ```sh
  python main.py
