# Final Project

## Project Description

We have used 2 different classification methods to predict 5 classes of heart disease. We have used a decion tree classifier and a gradient descent classifier to classify [this](https://archive.ics.uci.edu/dataset/45/heart+disease) popular heart disease dataset. The dataset gives 14 attributes and classifies heart disease presence with an integer from 0 to 4. For more information on the dataset please navigate to 'heart+disease\heart-disease.names' for the documentation that came with the data.

After predicting heart disease from a test data set, the program will print 2 graphs to compare the Accuracies and ROC curves of each classification method.

## Quick Start

There is no required input for this .py file. Run the `main.py` file and it will create an ROC graph for the Decision Tree method. After closing that window, the program will display the ROC graph for the Gradient Descent Method. After closing that window, the program will terminate


# Dependencies 
## From /heart+disease/
The only data files that are used by the program are:
- /heart+disease/processed.cleveland.data
- /heart+disease/processed.hungarian.data
- /heart+disease/processed.switzerland.data
- /heart+disease/processed.va.data

## Packages:
- numpy
- pandas
- matplotlib
- traceback
- sklearn
- itertools

# Class Diagrams
## DecisionTreeModule.py
+-------------------+
|  DecisionTree     |
+-------------------+
| - cleveland       |
| - n_classes       |
| - X_train         |
| - X_test          |
| - y_train         |
| - y_test          |
| - y_score         |
+-------------------+
| + __init__()      |
| + load_data()     |
| + split_data()    |
| + fit()           |
| + graph_roc()     |
+-------------------+

## GradientDescentModule.py
+-------------------+
|  GradientDescent  |
+-------------------+
| - cleveland       |
| - n_classes       |
| - X_train         |
| - X_test          |
| - y_train         |
| - y_test          |
| - y_score         |
+-------------------+
| + __init__()      |
| + load_data()     |
| + split_data()    |
| + fit()           |
| + graph_roc()     |
+-------------------+
