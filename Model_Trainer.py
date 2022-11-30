# The class trains a model on doodle images.
# the images are preprocessed before training 
# with a SVM classifier.
# it starts by preprocessing the images and then generates
# new images using affine transformation.
# it then trains the SVM classifier and saves the model
# to the disk.
from SVM_Classification import SVM_Classifier
import os
import sys

class Model_Trainer:
    def __init__(self,train_data_path, model_path, model_name, num_components, kernel_type, C):
        self.train_data_path = train_data_path
        self.model_path = model_path
        self.model_name = model_name
        self.num_components = num_components
        self.kernel_type = kernel_type
        self.C = C
        self.svm = SVM_Classifier(self.train_data_path, self.model_path, self.model_name, self.num_components, self.kernel_type, self.C)

        self.X, self.y = self.svm.load_data()
        self.model = None

        self.val_accuracy = None
        self.val_std = None

        self.test_accuracy = None
        self.test_cm = None
        self.test_cr = None


    def train_model(self):
        #train SVM classifier
        self.model = self.svm.train(self.X, self.y)

    def cross_validation(self):
        #cross validation
        self.val_accuracy, self.val_std = self.svm.cross_validation(self.X, self.y, self.model)
        print("accuracy: ", self.val_accuracy)
        print("standard deviation: ", self.val_std)

    def test_model(self):
        #test SVM classifier
        self.test_accuracy, self.test_cm, self.test_cr = self.svm.test(self.X, self.y, self.model)
        print("accuracy: ", self.test_accuracy)
        print("confusion matrix: ", self.test_cm)
        print("classification report: ", self.test_cr)
    
    def main(self):
        self.train_model()
        self.cross_validation()















