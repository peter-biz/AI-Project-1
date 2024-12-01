# Alec Malenfant
# Peter Bizoukas
# This script is the driver for the project. Run this file

from DecisionTreeModule import DecisionTree
from GradientDescentModule import GradientDescent
import os

if __name__ == '__main__':
    # Change working directory to the path of this python file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    dt = DecisionTree()  # Create object
    gd = GradientDescent()  # Create Object
