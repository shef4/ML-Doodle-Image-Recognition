#q: how to import the class from the other file?
from models.SVM_Classifier import SVM_Classifier
from models.KNN_Classifier import KNN_Classifier
from models.CNN_Classifier import CNN_Classifier

class Model_Trainer:
    #TODO: add CNN and KNN hyperparameters at end with default values
    def __init__(self, train_data_path, model_type, model_path, model_name, num_components = None, kernel_type = None, C = None):
        self.train_data_path = train_data_path
        self.model_type = model_type
        self.model_path = model_path
        self.model_name = model_name
        self.num_components = num_components
        self.kernel_type = kernel_type
        self.C = C

        #TODO: add CNN and KNN classifiers
        if self.model_type == "SVM":
            self.clf = SVM_Classifier(self.train_data_path, self.model_path, self.model_name, self.num_components, self.kernel_type, self.C)
        if self.model_type == "CNN":
            self.clf = CNN_Classifier(self.train_data_path, self.model_path, self.model_name)
        if self.model_type == "KNN":
            self.clf = KNN_Classifier(self.train_data_path, self.model_path, self.model_name)
        else:
            print("Invalid model type")

        self.X, self.y = self.clf.load_data()

        self.model = None

        self.val_accuracy = None
        self.val_std = None

        self.test_accuracy = None
        self.test_cm = None
        self.test_cr = None


    def train_model(self):
        #train SVM classifier
        #TODO: implement CNN and KNN train() functionality in models folder
        self.model = self.clf.train(self.X, self.y)

    def cross_validation(self):
        #cross validation
        #TODO: implement CNN and KNN cross_validation() functionality in models folder
        self.val_accuracy, self.val_std = self.clf.cross_validation(self.X, self.y, self.model)
        print("accuracy: ", self.val_accuracy)
        print("standard deviation: ", self.val_std)

    def test_model(self):
        #test classifier
        #TODO: implement CNN and KNN test() functionality in models folder
        self.test_accuracy, self.test_cm, self.test_cr = self.clf.test(self.X, self.y, self.model)
        print("accuracy: ", self.test_accuracy)
        print("confusion matrix: ", self.test_cm)
        print("classification report: ", self.test_cr)
    
    def main(self):
        self.train_model()
        self.cross_validation()















