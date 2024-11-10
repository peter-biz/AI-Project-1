# Alec Malenfant
# Peter Bizoukas
# This Python Script will predict heart disease using decision trees

# Right now it is only using the Cleveland data

import numpy as np
import pandas as pd
import matplotlib as plt
import traceback
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class DecisionTree:

    def __init__(self):
        self.cleveland = self.load_data()  # Load Cleveland Data
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()  # Split into testing and training data
        self.score = 0.0  # Accuracy Score
        self.fit_and_predict()  # Train model and predict heart disease


        return

    # Import Data
    def load_data(self):
        # Load the Cleveland dataset
        try:
            cleveland = pd.read_csv('./heart+disease/processed.cleveland.data')
            return cleveland
        except FileNotFoundError:
            print("Error : './heart+disease/processed.cleveland.data' not found")
            print(traceback.format_exc())
        except Exception as e:
            print(traceback.format_exc())

    # Split Data
    def split_data(self):
        # remove rows with missing data
        self.cleveland = self.cleveland.replace(to_replace='?', value=np.nan)  # Replace question marks with null values
        self.cleveland = self.cleveland.dropna()  # Remove all rows with null values

        X = self.cleveland.iloc[:, :-1]  # data
        y = self.cleveland.iloc[:, -1]  # labels

        # Split data into testing and training
        return train_test_split(X, y, random_state=0, test_size=0.8)


    # Train Model and Predict Data
    def fit_and_predict(self):
        # Train decision tree model
        dtc = DecisionTreeClassifier()
        dtc.fit(self.X_train, self.y_train)

        # Predict Heart Disease
        dtc.predict(self.X_test)

        # Evaluate Accuracy
        self.score = dtc.score(self.X_test, self.y_test)
        print("Score : " + str(self.score))




if __name__ == '__main__':
    decisionTree = DecisionTree()  # Create object

