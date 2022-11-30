#class to pre process doodle images for SVM classification
import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt
from Image_Generator import Image_Generator

class Preprocess_Image:
    def __init__(self, src_image_path, processed_image_path, processed_image_name, gen_image_path, gen_image_name, num_images):
        self.src_image_path = src_image_path
        self.processed_image_path = processed_image_path
        self.processed_image_name = processed_image_name
        self.gen_image_path = gen_image_path
        self.gen_image_name = gen_image_name
        self.num_images = num_images

    def load_data(self, src_image_path):
        #load data from data_path
        #return X and y
        X = []
        y = []
        for root, dirs, files in os.walk(src_image_path):
            for file in files:
                if file.endswith(".jpg"):
                    img = cv2.imread(os.path.join(root, file), 0)
                    img = img.flatten()
                    X.append(img)
                    y.append(root.split("/")[-1])
        X = np.array(X)
        y = np.array(y)
        return X, y
    
    
    def generate_images(self):
        #generate new images
        for image in os.listdir(self.src_image_path):
            image_path = os.path.join(self.src_image_path, image)
            image_generator = Image_Generator(image_path, self.gen_image_path, self.gen_image_name, self.num_images)
            image_generator.generate_images()


    def preprocess_images(self, X, y):
        #preprocess images
        #return X and y
        X = self.resize_images(X)
        X = self.filter_images(X)
        X = self.normalize_images(X)
        self.save_images(X, y)
        return X, y

    def resize_images(self, X):
        #resize images
        #return X
        X = np.array([cv2.resize(img.reshape(28, 28), (28, 28)) for img in X])
        return X

    def filter_images(self, X):
        #filter images for better image classification
        #return X
        X = np.array([cv2.bilateralFilter(img.reshape(28, 28), 9, 75, 75) for img in X])
        return X

    def normalize_images(self, X):
        #normalize images
        #return X
        X = X / 255.0
        return X

    def save_images(self, X, y):
        #save images add number of images currently in file to image name
        #return None
        for i in range(len(X)):
            #get the nunmber of images in the folder
            num_images = len(os.listdir(os.path.join(self.processed_image_path, y[i])))
            img = X[i].reshape(28, 28)
            cv2.imwrite(os.path.join(self.processed_image_path, y[i], self.processed_image_name + str(num_images) + ".jpg"), img)
    
    def plot_image(self, X, y):
        #plot image
        #return None
        for i in range(len(X)):
            img = X[i].reshape(28, 28)
            plt.imshow(img, cmap='gray')
            plt.title(y[i])
            plt.show()
            break

    def split_data(self, src_data_path, train_data_path, test_data_path):
        #split data into 80% train, 20% test
        #return None
        for root, dirs, files in os.walk(src_data_path):
            for file in files:
                if file.endswith(".jpg"):
                    img = cv2.imread(os.path.join(root, file), 0)
                    img = img.flatten()
                    if np.random.rand() < 0.8:
                        cv2.imwrite(os.path.join(train_data_path, root.split("/")[-1], file), img)
                    else:
                        cv2.imwrite(os.path.join(test_data_path, root.split("/")[-1], file), img)

    def preprocess(self):
        #preprocess data
        #return None
        X, y = self.load_data(self.src_image_path)
        X, y = self.preprocess_images(X, y)
        self.generate_images()
        X, y = self.load_data(self.gen_image_path)
        X, y = self.preprocess_images(X, y)
        self.split_data(self.processed_image_path, "data/dev/train", "data/test")

if __name__ == "__main__":
    src_image_path = "data/raw"
    processed_image_path = "data/preprocessed/processed"
    processed_image_name = "processed_image"
    gen_image_path = "data/preprocessed/generated"
    gen_image_name = "gen_image"
    num_images = 10
    preprocess_image = Preprocess_Image(src_image_path, processed_image_path, processed_image_name, gen_image_path, gen_image_name, num_images)
    preprocess_image.preprocess()
