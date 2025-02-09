# -*- coding: utf-8 -*-
"""
This program does the followings: 
    1- To perform model-dependant artificail neural network (ANN) preprocessing (inclduing pixel conversion);
    2- To find the best hyperparameter parameters for a multi-layer normal ANN; and
    3- Run the model for over a 5-fold cross-validation

@author: MahdaviAl
"""

import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import KFold
from keras import layers
import keras_tuner as kt
from math import floor
from sklearn.model_selection import train_test_split
from image_utils import shuffle_data, augment_dataset, get_normalized_counts
import h5py
import os
import numpy as np


#########################################################
### Step 1: Data augmentation over imbalanced classes ###
#########################################################

# Read feature image and target labels
save_folder_array = r'C:\python_projects\cnn\for_image_classifier\processed'
with h5py.File(os.path.join(save_folder_array, 'dataset.h5'), 'r') as hf:
    X = hf['X'][:]  # Read the entire 'X' dataset
    y = hf['y'][:]  # Read the entire 'y' dataset


# Augment data
classes_to_augment = {'drawing', 'signature', 'logo'}
X, y = augment_dataset(X, y, classes_to_augment, crop_ratio = 0.8)


##############################################################
### Step 2: Model-dependant pre-processsing & verification ###
##############################################################

# Display the image
index = 500  # index of the element to display
plt.imshow(X[index], cmap = 'gray')
plt.axis('off')  # Turn off axis labels
plt.show()

# Shuffle data for proper train test split
X, y = shuffle_data(X, y)


# Transform all pixels to the range [-1, 1]
X = (X / 127.5) - 1.0

# Split to train and test (manually)
threshold = floor(0.8 * len(y)) + 1
X_train, X_serv, y_train, y_serv = X[:threshold], X[threshold+1:], y[:threshold], y[threshold+1:]


# Verify if shuffle results in equal number in sample splits
train_df = get_normalized_counts(y_train, 'train')
test_df = get_normalized_counts(y_serv, 'serv')
distributed_class = pd.concat([train_df, test_df], axis = 1)
print(distributed_class)


#######################################################
### Step 3: Keras tuner and NAS to find best HP set ###
#######################################################

# Define pre-training variables
batch_size = 64
target_size = X.shape[1]

# Define a hypermodel for ANN architecture search
def build_ann_model(hp):
    model = keras.Sequential()

    # Input layer for 512x512 images (RGB or grayscale)
    model.add(layers.Flatten(input_shape = (target_size, target_size, 3)))  # Adjust for grayscale if needed

    # Tune the number of hidden layers and units
    for i in range(hp.Int('num_layers', 1, 10)):
        model.add(layers.Dense(
            units=hp.Int(f'units_{i}', min_value = 8, max_value = 512, step = 32),
            activation=hp.Choice('activation', values = ['relu', 'tanh', 'sigmoid'])
        ))

    # Optional: Add dropout for regularization
    model.add(layers.Dropout(rate=hp.Float('dropout', min_value = 0.1, max_value = 0.5, step = 0.1)))

    # Output layer
    model.add(layers.Dense(units = 4, activation = 'softmax'))  # Example: 10-class classification

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values = [1e-3, 1e-4, 1e-5])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Initialize the tuner
tuner = kt.RandomSearch(
    build_ann_model,
    objective = 'val_accuracy',
    max_trials = 10,                
    executions_per_trial = 1,
    directory = 'my_tuner_results',
    project_name = 'kfold_ann_tuning'
)


# Use a simple 80-20 train-validation split
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size = 0.2, random_state = 42
)

# Run the tuner and dynamically tune batch size
tuner.search(
    X_train_split,
    y_train_split,
    validation_data = (X_val_split, y_val_split),
    epochs = 20,
    batch_size = batch_size
)

# Get the best hyperparameters after cross-validation
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

# Print the best hyperparameters
print("Best hyperparameters:")
print(f"Number of layers: {best_hps.get('num_layers')}")
print(f"Activation: {best_hps.get('activation')}")
print(f"Units per layer: {[best_hps.get(f'units_{i}') for i in range(best_hps.get('num_layers'))]}")
print(f"Dropout rate: {best_hps.get('dropout')}")
print(f"Learning rate: {best_hps.get('learning_rate')}")


#############################
### Step 4: Model Running ###
#############################

# Function to build the final model
def build_final_model():
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=(target_size, target_size, 3)))

    # Apply the best hyperparameters
    for i in range(best_hps['num_layers']):
        model.add(layers.Dense(
            units=best_hps[f'units_{i}'],
            activation=best_hps['activation']
        ))

    # Add dropout and output layer
    model.add(layers.Dropout(rate = best_hps['dropout']))
    model.add(layers.Dense(units = 10, activation = 'softmax'))

    # Compile the model
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate = best_hps['learning_rate']),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )
    return model

# Initialize K-fold cross-validator
kf = KFold(n_splits = 5, shuffle = True, random_state = 42)

# Cross-validation training and evaluation
cv_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"Training fold {fold + 1}...")

    # Split the data for this fold
    x_train_fold, x_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

    # Build and train the model
    model = build_final_model()
    model.fit(
        x_train_fold, y_train_fold,
        epochs = 10,
        batch_size = best_hps.get('batch_size'),
        validation_data = (x_val_fold, y_val_fold),
        verbose=1
    )

    # Evaluate on the validation set
    val_loss, val_accuracy = model.evaluate(x_val_fold, y_val_fold, verbose = 0)
    cv_scores.append(val_accuracy)

# Calculate the average cross-validation accuracy
avg_cv_accuracy = np.mean(cv_scores)
print(f"Average CV accuracy: {avg_cv_accuracy:.2f}")

# Train the final model on the entire training set
final_model = build_final_model()
final_model.fit(X_train, y_train, epochs = 20, batch_size = best_hps.get('batch_size'), verbose = 1)

# Evaluate on the test set
test_loss, test_accuracy = final_model.evaluate(X_serv, y_serv)
print(f"Test accuracy: {test_accuracy:.2f}")

