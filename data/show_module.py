# import all libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle 
import datetime as dt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.svm import SVC
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
        #changing Scheduled_day and Appointment_day datatype to timestamps
        df['Scheduled_day'] = pd.to_datetime(df['Scheduled_day'].dt.date)
        df['Appointment_day'] = pd.to_datetime(df['Appointment_day'].dt.date)
        # Extracting the 'Year','Month' and 'day' from Date Column.
        df['scheduled_month'] = df['Scheduled_day'].dt.month_name()
        df['scheduled_day'] = df['Scheduled_day'].dt.day_name()

        df['appointment_month'] = df['Appointment_day'].dt.month_name()
        df['appointment_day'] = df['Appointment_day'].dt.day_name()

        # converting the month and day columns to interger values as this is a ML project
        df['scheduled_month'] = df['scheduled_month'].replace(['January', 'February', 'March', 'April', 'May', 'June', 
        'July', 'August', 'September', 'October', 'November', 'December'], ['1', '2', '3', '4', '5', '6', '7', '8', '9', 
        '10', '11', '12'])
        df['scheduled_day'] = df['scheduled_day'].replace(['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday','Friday', 
        'Saturday'], ['1', '2', '3', '4', '5', '6', '7'])

        df['appointment_month'] = df['appointment_month'].replace(['January', 'February', 'March', 'April', 'May', 'June', 
        'July', 'August', 'September', 'October', 'November', 'December'], ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
        '11', '12'])
        df['appointment_day'] = df['appointment_day'].replace(['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday','Friday', 
        'Saturday'], ['1', '2', '3', '4', '5', '6', '7'])
        # to remove the irrelevant columns
        df = df.drop(['Appointment_day', 'Scheduled_day'], axis = 1, inplace = True)

        # changing strings value ('yes','no') in 'No_show to int values ('0','1')
        # with this, every '1' value in the 'show' dataset indicates that the patient showed up
        # and '0' means the patient didnt.
        df['Show'].mask(df['Show'] == 'No', 1, inplace=True)
        df['Show'].mask(df['Show'] == 'Yes', 0, inplace=True)
        # changing the datatype of 'show' to reflect it's new content and ease computation
        df.Show = df.Show.astype(int)
        # to remove the irrelevant columns
        df = df.drop(['Patient_ID', 'Appointment_ID', 'Appointment_day', 'Scheduled_day'], axis = 1, inplace = True)
        
    # a function which outputs the probability of a data point to be 1
    def predicted_probability(self):
        if (self.data is not None):
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs

    # predict the outputs and the probabilities and add columns with these values at the end of new data
    def predicted_outputs(self):
        if (self.data is not None):
            self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
            self.preprocessed_data['Prediction'] = self.reg.predict(self.data)
            return self.preprocessed_data

