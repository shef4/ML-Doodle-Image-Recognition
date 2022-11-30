#class to generate new doodle images from a given image
import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

class Image_Generator:
    def __init__(self, src_image_path, gen_image_path, gen_image_name, num_images):
        self.src_image_path = src_image_path
        self.gen_image_path = gen_image_path
        self.gen_image_name = gen_image_name
        self.num_images = num_images

    def affine_transform(self, image):
        #generate new image using affine transformation
        #return new image
        image = img_to_array(image)
        image = image.reshape((1,) + image.shape)
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
        i = 0
        for batch in datagen.flow(image, batch_size=1, save_to_dir=self.gen_image_path, save_prefix=self.gen_image_name, save_format='jpg'):
            i += 1
            if i > self.num_images:
                break

    def load_image(self):
        #load image from curr_image_path
        #return image
        image = load_img(self.src_image_path)
        return image

    def generate_images(self):
        #generate new images
        image = self.load_image()
        self.affine_transform(image)


    

