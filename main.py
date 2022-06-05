#Import necessary libraries
from data_preprocessing import *
from cnn_model import *
import pandas as pd
import numpy as np
import cv2
import time
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# Disable debugging logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Load data
col_names = ['center_camera', 'left_camera', 'right_camera', 'steering_angle', 'throttle', 'brake', 'speed']
df = pd.read_csv('data/driving_log.csv', names = col_names)

print('Total Images Imported: ', df.shape[0])

# Save relevant data for our model
images_path = df['center_camera'].values
steering_angles = df['steering_angle'].values

# Divide data into training and validation (80/20)
train_df, val_df = train_test_split(df, test_size = 0.2)
print('Total Training Images: ', len(train_df))
print('Total Validation Images: ', len(val_df), '\n')
    
# Load the model
cnn_model = nvidia_model()
cnn_model.summary()
print('\n')

# Fit the model
print('Training the Model...')
start_time = time.time()

cnn_model.fit(batch_generator(train_df, batch_size = 100, training_flag = 1),
              steps_per_epoch = 300, epochs = 10, 
              validation_data = batch_generator(val_df, batch_size = 100, training_flag = 0),
              validation_steps = 300)

end_time = time.time()
training_time = end_time - start_time
print('\n', 'Training Duration: ', training_time)
    
# Save the model
cnn_model.save('model.h5')
print('Model Saved')

