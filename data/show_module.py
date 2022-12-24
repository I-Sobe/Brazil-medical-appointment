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
        labels =['Patient_ID', 'Appointment_ID', 'Gender', 'Scheduled_day', 'Appointment_day', 'Age', 'Neighbourhood', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received', 'Show']
        df = pd.read_csv(data_file, delimiter=',', header=0, names=labels)
        # store the data in a new variable for later use
        self.df_with_predict = df.copy()
        # to remove the irrelevant columns
        df = df.drop(['Patient_ID', 'Appointment_ID'], axis = 1, inplace = True)
        #changing Scheduled_day and Appointment_day datatype to timestamps
        df['Scheduled_day'] = pd.to_datetime(df['Scheduled_day'])
        df['Appointment_day'] = pd.to_datetime(df['Appointment_day'])
