#class for SVM doodle image classification
import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class SVM_Classifier:
    def __init__(self, data_path, model_path, model_name, num_components, kernel_type, C):
        self.data_path = data_path
        self.model_path = model_path
        self.model_name = model_name
        self.num_components = num_components
        self.kernel_type = kernel_type
        self.C = C
        
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
        #train SVM model
        #return trained model
        X, y = self.load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        pca = PCA(n_components=self.num_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        model = svm.SVC(kernel=self.kernel_type, C=self.C)
        model.fit(X_train, y_train)
        self.save_model(model)
        y_pred = model.predict(X_test)
        print("Accuracy: ", accuracy_score(y_test, y_pred))
        print("Confusion Matrix: ")
        print(confusion_matrix(y_test, y_pred))
        print("Classification Report: ")
        print(classification_report(y_test, y_pred))
        return model

    def test(self):
        #test SVM model
        #return accuracy, confusion matrix, classification report
        X, y = self.load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        pca = PCA(n_components=self.num_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        model = self.load_model()
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        return accuracy, cm, cr

    def cross_validation(self):
        #cross validation
        #return accuracy, confusion matrix, classification report
        X, y = self.load_data()
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        pca = PCA(n_components=self.num_components)
        X = pca.fit_transform(X)
        model = svm.SVC(kernel=self.kernel_type, C=self.C)
        kfold = KFold(n_splits=10, random_state=0)
        results = cross_val_score(model, X, y, cv=kfold)
        print("Accuracy: ", results.mean())
        print("Standard Deviation: ", results.std())
        return results.mean(), results.std()

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


