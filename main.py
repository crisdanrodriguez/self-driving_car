#Import necessary libraries
from data_augmentation import *
from cnn_model import *
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def generator(df, batch_size = 32):
    # Arrays to store images and steering angles
    batch_images = np.zeros((batch_size, 66, 200, 3))
    batch_steering_angles = np.zeros(batch_size)

    # Generate more images from one image using data_augmentation
    while True:
        for i_batch in range(batch_size):
            i_row = np.random.randint(len(df))
            row_data = df.iloc[[i_row]].reset_index()

            image, steering_angle = augment_image(row_data)

            # Store image and steering angle
            batch_images[i_batch] = image
            batch_steering_angles[i_batch] = steering_angle
            
        yield batch_images, batch_steering_angles


# Load data
df = pd.read_csv('data/driving_log.csv')

# Divide data into training and testing (80/20)
train_df, test_df = train_test_split(df, test_size = 0.2)

# Create training data for each epoch
train_generator = generator(train_df, batch_size = 64)
# Create testing data for each epoch
test_generator = generator(test_df, batch_size = 64)
    
# Fit the model
cnn_model = nvidia_model()
cnn_model.fit(train_generator, steps_per_epoch = 20000, epochs = 5, validation_data = test_generator)
    
# Save the model
cnn_model.save('model.h5')

print('Model Saved')

# Print model summary
cnn_model.summary()