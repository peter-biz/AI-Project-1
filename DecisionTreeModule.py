# Alec Malenfant
# Peter Bizoukas
# This Python Script will predict heart disease using decision trees

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import label_binarize, normalize
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
np.set_printoptions(threshold=np.inf)  # Comment out to make printed matrices truncated

class DecisionTree:

    def __init__(self):
        #__ Driver __

        self.cleveland = self.load_data()  # Load Cleveland Data
        self.n_classes = 0  # Number of classes in data. Assigned during split_data
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()  # Split into testing and training data
        self.y_score = self.fit()  # Train model and get score
        self.graph_roc()  # graph ROC for all 4 classes


        return

    # Import Data
    def load_data(self):
        # Load the Cleveland dataset
        try:
            cleveland = pd.read_csv('./heart+disease/processed.cleveland.data')
            hungary = pd.read_csv('./heart+disease/processed.hungarian.data')
            switzerland = pd.read_csv('./heart+disease/processed.switzerland.data')
            va = pd.read_csv('./heart+disease/processed.va.data')
            combined_data = pd.concat([cleveland, hungary, switzerland, va])
            return combined_data
        except FileNotFoundError:
            print("Error : './heart+disease/processed.cleveland.data' not found")
            print(traceback.format_exc())
        except Exception as e:
            print(traceback.format_exc())

    # Split Data
    def split_data(self):
        # remove rows with missing data
        self.cleveland = self.cleveland.replace(to_replace='?', value=np.nan)  # Replace question marks with null values
        self.cleveland = self.cleveland.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric, coercing errors to NaN
        for column in self.cleveland.columns:
            self.cleveland[column] = self.cleveland[column].fillna(self.cleveland[column].median())  # Replace NA with median of the respective column

        # Split data into features and labels
        X = self.cleveland.iloc[:, :-1]  # features
        y = self.cleveland.iloc[:, -1]  # labels

        # Normalize Data with l2 regularization
        X = normalize(X, "l2")

        # Binarize the output
        y = label_binarize(y, classes=[0,1,2,3,4])
        self.n_classes = y.shape[1]

        # Split data into testing and training
        return train_test_split(X, y, random_state=0, test_size=0.8)

    # Train Model and Predict Data
    def fit(self):
        self.classifier = OneVsRestClassifier(DecisionTreeClassifier(random_state=255))  # Create classifier object
        return self.classifier.fit(self.X_train, self.y_train).predict_proba(self.X_test)  # Train and test data

    # Create ROC Graph
    def graph_roc(self):
        fpr = dict()  # False Positive Rate
        tpr = dict()  # True Positive Rate
        roc_auc = dict()
        
        # Calculate ROC curve and AUC for each class
        for i in range(self.n_classes):
            fpr[i], tpr[i], _ = roc_curve(self.y_test[:, i], self.y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Class colors
        colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])
        
        # Plot each point in each class on the ROC
        for i, color in zip(range(self.n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Heart Disease Data by Class \nDecision Tree Method')
        plt.legend(loc="lower right")
        plt.show()

        return
