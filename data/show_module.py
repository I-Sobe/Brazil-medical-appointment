# import all libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

# creating the special class fo the prediction
class appointment_model():

    def __init__(self, model_file, scalar_file):
        # reads the 'model' and 'scalar' files
        with open('model', 'rb') as model_file, open('scaler', 'rb') as scaler_file:
            self.classifier = pickle.load(model_file)
            self.scaler = pickle.load(scalar_file)
            self.data = None

    # take a data file (.csv) and preprocessit in the same way in the preprocessing file
    def load_and_clean_data(self, data_file):

        # import the data
        df = pd.read_csv(data_file, delimiter=',')
        # store the data in a ne