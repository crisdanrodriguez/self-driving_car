'''
This file contains all the necessary functions for image augmentation
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


def brightness_modification(image):
    # Convert the image from RGB to HSV 
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Convert pixels dtype from uint8 to float64 
    hsv_image = np.array(hsv_image, dtype = np.float64)
    
    # Modify the brigthness changing the V value
    random_brightness = 0.7 + np.random.uniform()
    hsv_image[:,:,2] = hsv_image[:,:,2] * random_brightness
    hsv_image[:,:,2][hsv_image[:,:,2] > 255] = 255
    
    # Return pixels dtype to uint8 
    hsv_image = np.array(hsv_image, dtype = np.uint8)
    
    # Returns the image to RGB
    rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    
    return rgb_image


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
    # Random steering angle calibration
    angle_calibration_p = np.random.randint(3)
    if angle_calibration_p == 0:
        image_path = df['center_camera'][0].strip()
        calibration_angle = 0
    elif angle_calibration_p == 1:
        image_path = df['left_camera'][0].strip()
        calibration_angle = 0.25
    elif angle_calibration_p == 2:
        image_path = df['right_camera'][0].strip()
        calibration_angle = -0.25
    steering_angle = df['steering_angle'][0] + calibration_angle

    # Read image
    image = cv2.imread(image_path)

    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

    # Horizontal and vertical shifts
    image, steering_angle = translation(image, steering_angle)

    # Brightness augmentation
    image = brightness_modification(image)

    # Image cropping
    image = top_bottom_crop(image)

    # Resize image
    image = cv2.resize(image, (66, 200), interpolation = cv2.INTER_AREA)

    # Horizontal flip
    horizontal_flip_p = np.random.randint(2)
    if horizontal_flip_p == 0:
        image, steering_angle = horizontal_flip(image, steering_angle)

    return image, steering_angle


def image_generator(df, batch_size = 32):
    # Arrays to store images and steering angles
    batch_images = np.zeros((batch_size, 66, 200, 3))
    batch_steering_angles = np.zeros(batch_size)

    # Generate more images from one image using data_augmentation
    while True:
        for i_batch in range(batch_size):
            i_row = np.random.randint(len(df))
            row_data = df.iloc[[i_row]].reset_index()

            keep_probability = 0

            # Images with lower angles have lower probability of getting represented in the data set
            while keep_probability == 0:
                image, steering_angle = augment_image(row_data)
                if abs(steering_angle) < 0.1:
                    pr_val = np.random.uniform()
                    if pr_val > pr_threshold:
                        keep_probability = 1
                else:
                    keep_probability = 1

            batch_images[i_batch] = image
            batch_steering_angles[i_batch] = steering_angle
            
        yield batch_images, batch_steering_angles




