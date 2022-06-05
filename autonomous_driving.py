'''
This file is for testing the model in the Udacity simulator and was taken from:
https://youtu.be/mVUrErF5xq8

Author: Murtaza's Workshop - Robotics and AI
Date: July 7, 2020
'''

print('Setting UP')

# Import necessary libraries
from data_preprocessing import *
import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
import os

# Disable debugging logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


sio = socketio.Server()

app = Flask(__name__) # '__main__'
maxSpeed = 30


@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = image_preprocessing(image)
    image = np.array([image])
    steering = float(model.predict(image))
    throttle = 1.0 - speed / maxSpeed
    print('{} {} {}'.format(steering, throttle, speed))
    sendControl(steering, throttle)


@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    sendControl(0, 0)


def sendControl(steering, throttle):
    sio.emit('steer', data = {
        'steering_angle': steering.__str__(),
        'throttle': throttle.__str__()
    })


if __name__ == '__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)