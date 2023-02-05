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
        labels =['patient_ID', 'appointment_ID', 'gender', 'Scheduled_day', 'Appointment_day', 'age', 'neigbhorhood', 'scholarship', 'hypertension', 'diabetes', 'alcoholism', 'handicap', 'sms_received', 'show']
        df = pd.read_csv(data_file, delimiter=',', header=0, names=labels)
       
        # store the data in a new variable for later use
        self.df_with_predict = df.copy()
        
        #changing Scheduled_day and Appointment_day datatype to timestamps
        df['Scheduled_day'] = pd.to_datetime(df['scheduled_day'].dt.date)
        df['Appointment_day'] = pd.to_datetime(df['appointment_day'].dt.date)
        # Extracting the 'Year','Month' and 'day' from Date Column.
        df['scheduled_month'] = df['Scheduled_day'].dt.month_name()
        df['scheduled_day'] = df['Scheduled_day'].dt.day_name()

        df['appointment_month'] = df['Appointment_day'].dt.month_name()
        df['appointment_day'] = df['Appointment_day'].dt.day_name()

        # converting the month and day columns to interger values as this is a ML project
        df.scheduled_month.replace(to_replace=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 
        'September', 'October', 'November', 'December'], value=[1,2,3,4,5,6,7,8,9,10,11,12], inplace=True)
        df.scheduled_day.replace(to_replace=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday','Friday', 'Saturday'],
        value=['1', '2', '3', '4', '5', '6', '7'], inplace=True)
        df.appointment_month.replace(to_replace=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 
        'September', 'October', 'November', 'December'], value=[1,2,3,4,5,6,7,8,9,10,11,12], inplace=True)
        df.appointment_day.replace(to_replace=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday','Friday', 'Saturday'],
        value=['1', '2', '3', '4', '5', '6', '7'], inplace=True)

        # changing strings value ('Yes','No) in 'No_show to int values ('0', '1')
        df['Show'].replace(to_replace=['Yes', 'No'], value=[0,1], inplace=True)
        # changing the datatype of 'show' to reflect it's new content and ease computation
        df.Show = df.Show.astype(int)
        # make sure the age column doesn't have a value less than 0
        df = df[df['Age'] >=0]
        
        # feature engineering the neigbhorhood column
        neigbhorhood = pd.get_dummies(df.neigbhorhood, drop_first=True)
        # grouping the neigbhorhood data
        neigbhorhood_1 = neigbhorhood.iloc[:, :11].max(axis=1)
        neigbhorhood_2 = neigbhorhood.iloc[:, 11:21].max(axis=1)
        neigbhorhood_3 = neigbhorhood.iloc[:, 21:31].max(axis=1)
        neigbhorhood_4 = neigbhorhood.iloc[:, 31:41].max(axis=1)
        neigbhorhood_5 = neigbhorhood.iloc[:, 41:51].max(axis=1)
        neigbhorhood_6 = neigbhorhood.iloc[:, 51:61].max(axis=1)
        neigbhorhood_7 = neigbhorhood.iloc[:, 61:71].max(axis=1)
        neigbhorhood_8 = neigbhorhood.iloc[:, 71:].max(axis=1)
        # concatenate the columns
        df = pd.concat([df, df, neigbhorhood_1, neigbhorhood_2, neigbhorhood_3, neigbhorhood_4, neigbhorhood_5, neigbhorhood_6, neigbhorhood_7, neigbhorhood_8], axis=1)
        
        # let's create age groups
        #create the bin_edges that will be used to cut the data into groups.
        bin_edges = [-1.0, 22, 38.0, 54.0, 115.0]

        #create labels for the new categories.
        # 1 (Gen_Z+), 2 (Milennials), 3 (Gen_X), 4 (Bloomers+)
        bin_names = ['1', '2', '3', '4']

        # puting the pandas_cut function to use
        df['age_groups'] = pd.cut(df['Age'], bin_edges, labels=bin_names)

        # to remove the irrelevant columns
        df = df.drop(['Patient_ID', 'Appointment_ID', 'Appointment_day', 'Scheduled_day', 'neigbhorhood', 'age',
        'scholarship', 'alcoholism', 'sms_received', 'neigbhorhood_1', 'neigbhorhood_3', 'neigbhorhood_4', 'neigbhorhood_6'], axis = 1, inplace = True)
        
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

