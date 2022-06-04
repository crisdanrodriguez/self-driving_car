'''
This file contains the function of the model

The CNN model architecture can be found at:
https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
'''

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


def nvidia_model():
    model = models.Sequential()

    # Layer 1
    model.add(layers.Conv2D(24, (5, 5), activation = 'elu', strides = (2, 2), input_shape = (66, 200, 3)))

    # Layer 2
    model.add(layers.Conv2D(36, (5, 5), activation = 'elu', strides = (2, 2)))

    # Layer 3
    model.add(layers.Conv2D(48, (5, 5), activation = 'elu', strides = (2, 2)))

    # Layer 4
    model.add(layers.Conv2D(64, (3, 3), activation = 'elu', strides = (1, 1)))

    # Layer 5
    model.add(layers.Conv2D(64, (3, 3), activation = 'elu', strides = (1, 1)))

    # Flatten
    model.add(layers.Flatten())

    # Full connected layer 1
    model.add(layers.Dense(100, activation = 'elu'))

    # Full connected layer 2
    model.add(layers.Dense(50, activation = 'elu'))

    # Full connected layer 3
    model.add(layers.Dense(10, activation = 'elu'))

    # Output layer
    model.add(layers.Dense(1))

    model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])

    return model