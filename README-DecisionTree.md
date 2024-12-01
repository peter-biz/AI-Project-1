# DecisionTreeModule.py
# Decision Tree
This file is run from the main.py file. It handles loading, splitting, training, and predicting data using a decision tree classifier. It also create an ROC graph.

## Requires the following packages:

- numpy
- pandas
- matplotlib
- traceback
- sklearn
- itertools

# Class Diagram
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
