import sklearn
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.metrics import max_error,mean_squared_error, mean_absolute_error, r2_score

import matplotlib.pyplot as plt
from math import sqrt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

import pandas as pd 
import numpy as np

def prepareData():
    data = pd.read_csv("steel.txt", sep="\t", header=None)
    data.columns = ["normalising_temperature","tempering_temperature", "sample" , "percent_silicon" , "percent_chromium", "manufacture_year", "percent_copper", "percent_nickel", "percent_sulphur", "percent_carbon", "percent_manganese", "tensile_strength"] 
    predict = "tensile_strength"

    #create list of data and corresponding list of variables to be predicted
    X = list(np.array(data.drop([predict], 1).drop(["sample"],1)))
    y = list(data[predict])
    
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)  
    
    return x_train, x_test, y_train, y_test

def svm(x_train,y_train):
    model = SVR(kernel='linear')
    model.fit(x_train,y_train)
    return model

def normalise():
    #TBC
    return null

def linearReg(x_train,y_train):
    model = linear_model.LinearRegression()
    model.fit(x_train,y_train)
    return model

def predict(model,x_test,y_test):
    print("----------------------------")
    accuracy = model.score(x_test,y_test)
    print(accuracy)
    
    predictions = model.predict(x_test)
    # The coefficients
    print('Coefficients: \n', model.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
        % mean_squared_error(y_test, predictions))
    # The Root mean squared error
    print("Root Mean squared error: %.2f"
        % sqrt(mean_squared_error(y_test, predictions)))
    # mean absolute error
    print("Mean absolute error: %.2f"
        % mean_absolute_error(y_test, predictions))
    # max absolute error
    print("Max absolute error: %.2f"
        % max_error(y_test, predictions))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, predictions))

    print("----------------------------")

x_train, x_test, y_train, y_test = prepareData()

linearModel = linearReg(x_train, y_train)
predict(linearModel,x_test,y_test)

svmModel = svm(x_train, y_train)
predict(svmModel,x_test,y_test)


