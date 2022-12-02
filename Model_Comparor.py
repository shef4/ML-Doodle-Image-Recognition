# The class calls the model triner class and trains various models
# to compare performance from different parameters.
# these parameters are:
# 1. number of components
# 3. kernel type
# 4. C value
from Model_Trainer import Model_Trainer
from matplotlib import pyplot as plt
import os
import sys

class Model_Comparor:
    def __init__(self, train_data_path, model_type, model_path, model_name,num_components, kernel_type, C):
        self.train_data_path = train_data_path
        self.model_type = model_type
        self.model_path = model_path
        self.model_name = model_name
        self.num_components = num_components
        self.kernel_type = kernel_type
        self.C = C
    
    def hyperparameter_accuracy(self, param_name):
        #compare model accuracy with different kernel types
        accuracy_array = []
        std_array = []
        start = 0
        end = 0
        steps = 0
        param = []

        if param_name == "num_components":
            start = 1
            end = 100
            steps = 10
        elif param_name == "kernel_type":
            param = ["linear", "poly", "rbf", "sigmoid"]
            start = 0
            end = 4
            steps = 1
        elif param_name == "C":
            start = 1
            end = 10
            steps = 1
        
        for i in range(start, end, steps):
            if param_name == "num_components":
                self.num_components = i
            elif param_name == "kernel_type":
                self.kernel_type = param[i]
            elif param_name == "C":
                self.C = i

            model_trainer = Model_Trainer(self.train_data_path, self.model_type, self.model_path, self.model_name, self.num_components, self.kernel_type, self.C)
            model_trainer.main()
            accuracy_array.append(model_trainer.val_accuracy)
            std_array.append(model_trainer.val_std)

        self.plot_accuracy(accuracy_array, param_name)
        self.plot_std(std_array, param_name)

    def plot_accuracy(self, accuracy_array, param_name):
        plt.plot(accuracy_array)
        plt.xlabel(param_name)
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs " + param_name)
        plt.show()

    def plot_std(self, std_array, param_name):
        plt.plot(std_array)
        plt.xlabel(param_name)
        plt.ylabel("Standard Deviation")
        plt.title("Standard Deviation vs " + param_name)
        plt.show()
    
    def run_comparisons(self):
        self.hyperparameter_accuracy("num_images")
        self.hyperparameter_accuracy("num_components")
        self.hyperparameter_accuracy("kernel_type")
        self.hyperparameter_accuracy("C")

if __name__ == "__main__":
    #TODO: change the model type and model name wbefore running
    model_type = "SVM"
    model_name = "model"

    #set memory path for model
    train_data_path = "data/dev/train"
    if model_type == "SVM":
        model_path = "models/svm_weights"
    elif model_type == "CNN":
        model_path = "models/cnn_weights"
    elif model_type == "KNN":
        model_path = "models/knn_weights"

    #set hyperparameters
    #SVM default parameters
    num_components = 10
    kernel_type = "linear"
    C = 1
    #CNN default parameters
    #TODO: add CNN default parameters
    #KNN default parameters
    #TODO: add KNN default parameters

    model_comparor = Model_Comparor(train_data_path, model_type, model_path, model_name, num_components, kernel_type, C)
    model_comparor.run_comparisons()













