'''
This file contains all the necessary functions for image preprocessing
'''

#Import necessary libraries
import numpy as np
import cv2


def horizontal_flip(image, steering_angle):
    # Flip the image horizontally
    flipped_image = cv2.flip(image, 1)
    
    # Reverse the sign of the steering angle
    steering_angle = -steering_angle
    
    return flipped_image, steering_angle


def brightness_reduction(image):
    # Convert the image from RGB to HSV 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Convert pixels dtype from uint8 to float64 
    image = np.array(image, dtype = np.float64)
    
    # Modify the brigthness changing the V value
    random_brightness = 0.8 - np.random.uniform(0, 0.4)
    image[:,:,2] = image[:,:,2] * random_brightness
    image[:,:,2][image[:,:,2] > 255] = 255
    
    # Return pixels dtype to uint8 
    image = np.array(image, dtype = np.uint8)
    
    # Returns the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    
    return image


def translation(image, steering_angle, x_translation_range = [-60, 60], y_translation_range = [-20, 20]):
    # Image dimensions
    height, width = (image.shape[0], image.shape[1])
    
    # Define translation along x and y axis
    x_translation = np.random.randint(x_translation_range[0], x_translation_range[1]) 
    y_translation = np.random.randint(y_translation_range[0], y_translation_range[1])
    
    # Adjust steering angle according to x_translation
    steering_angle += x_translation * 0.0035
    
    # Create translation matrix
    translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
    
    # Translate image uisng warpAffine
    translated_image = cv2.warpAffine(image, translation_matrix, (width, height))
    
    return translated_image, steering_angle


def top_bottom_crop(image):
    # Crop the bottom 25 and top 40 pixels
    cropped_image = image[40:135, :]
    
    return cropped_image


def augment_image(df):
    camera_side = np.random.randint(3)

    # Angle calibration according to image side
    if camera_side == 0:
        image_path = df.iloc[0]['center_camera'].strip()
        angle_calibration = 0
    elif camera_side == 1:
        image_path = df.iloc[0]['left_camera'].strip()
        angle_calibration = 0.25
    elif camera_side == 2:
        image_path = df.iloc[0]['right_camera'].strip()
        angle_calibration = -0.25

    steering_angle = df.iloc[0]['steering_angle'] + angle_calibration

    # Read the image as RGB
    image = cv2.imread(image_path)

    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

    if np.random.rand() < 0.5:
        # Horizontal and vertical shifts
        image, steering_angle = translation(image, steering_angle)

    if np.random.rand() < 0.5:
        # Brightness modification
        image = brightness_reduction(image)

    if np.random.rand() < 0.5:
        # Horizontal flip
        image, steering_angle = horizontal_flip(image, steering_angle)

    return image, steering_angle


def image_preprocessing(image):
    # Image cropping
    image = top_bottom_crop(image)

    # Image blurring
    image = cv2.GaussianBlur(image, (3,3), 0)

    # Resize image
    image = cv2.resize(image, (200, 66), interpolation = cv2.INTER_AREA)

    # Ranging pixel values from 0 to 1
    image = image / 255

    return image


def batch_generator(df, batch_size, training_flag):
    while True:
        # Lists for saving batch of images and steering angles
        images_bacth = []
        steering_angles_batch = []

        for i in range(batch_size):
            # Select a random row with image path and steering angle
            index = np.random.randint(0, len(df) - 1)

            # Just image augmentation for training data
            if training_flag:
                # Image augmentation
                image, steering_angle = augment_image(df.iloc[[index]])
            else:
                camera_side = np.random.randint(3)

                # Angle calibration according to image side
                if camera_side == 0:
                    image_path = df.iloc[0]['center_camera'].strip()
                    angle_calibration = 0
                elif camera_side == 1:
                    image_path = df.iloc[0]['left_camera'].strip()
                    angle_calibration = 0.25
                elif camera_side == 2:
                    image_path = df.iloc[0]['right_camera'].strip()
                    angle_calibration = -0.25

                #Read the image as RGB
                image = cv2.imread(image_path)
                steering_angle = df.iloc[0]['steering_angle'] + angle_calibration

            # Image preprocessing
            image = image_preprocessing(image)

            # Append image and steering angle
            images_bacth.append(image)
            steering_angles_batch.append(steering_angle)

        yield (np.asarray(images_bacth), np.asarray(steering_angles_batch))
