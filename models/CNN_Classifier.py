#class for KNN doodle image classification
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.externals import joblib

class CNN_Classifier:
    def __init__(self, data_path, model_path, model_name):
        self.data_path = data_path
        self.model_path = model_path
        self.model_name = model_name
        
    def load_data(self):
        #load data from data_path
        #return X and y
        X = []
        y = []
        #load .npy files from data_path
        for file in os.listdir(self.data_path):
            if file == "X_data.npy":
                X.append(np.load(os.path.join(self.data_path, file)))
            elif file == "Y_labels.npy":
                y.append(np.load(os.path.join(self.data_path, file)))
        X = np.array(X)
        y = np.array(y)
        return X, y
        
    def train(self):
        #TODO: train CNN model check SVM and modeltrainer for I/O datatypes and uses
        #return model
        pass

    def test(self):
        #TODO: test CNN modeL check SVM and modeltrainer for I/O datatypes and uses
        #       return accuracy, confusion matrix, classification report
        pass

    def cross_validation(self):
        # TODO: cross validation for CNN model check SVM and modeltrainer for I/O datatypes and uses
        #       return accuracy, std
        pass

    def load_model(self):
        #load model from model_path
        model = joblib.load(os.path.join(self.model_path, self.model_name))
        return model

    def save_model(self, model):
        #save model to model_path
        joblib.dump(model, os.path.join(self.model_path, self.model_name))

    def plot_accuracy(self, accuracy, title):
        #plot accuracy
        plt.plot(accuracy)
        plt.title(title)
        plt.ylabel('accuracy')
        plt.xlabel('iteration')
        plt.show()

    def plot_confusion_matrix(self, cm, title):
        #plot confusion matrix
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(cm))
        plt.xticks(tick_marks, cm, rotation=45)
        plt.yticks(tick_marks, cm)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
    
    def plot_classification_report(self, cr, title):
        #plot classification report
        lines = cr.split('  ')  
        classes, plotMat = [], []
        for line in lines[2 : (len(lines) - 3)]:
            t = line.strip().split(' ')
            classes.append(t[0])
            v = [float(x) for x in t[1:len(t) - 1]]
            plotMat.append(v)
        plt.imshow(plotMat, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        x_tick_marks = np.arange(3)
        class_names = ['precision', 'recall', 'f1-score']
        plt.xticks(x_tick_marks, class_names, rotation=45)
        y_tick_marks = np.arange(len(classes))
        plt.yticks(y_tick_marks, classes)
        plt.tight_layout()
        plt.ylabel('Classes')
        plt.xlabel('Measures')
        plt.show()  


