import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_regression
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import os.path
import re  

"""

This file calls inbuilt functions from a range of different feature selection related libraries and modules in order to output a set of features
deriving from the Z-Alizadeh data set.

Robustness: Both functions first check for the existence of an appropriate csv file which will contain the
dataset intended for the features to be selected from. If this file does not exist the functions will exit.


Scalability: These functions will work with other databases of larger sizes as long as they are properly formatted 
and the right target feature is specified within the code. 

"""
    


def chiSquared():

    """
    Funtion calls feature selection methods in the using a chiSquared module to determine the best features using the chi
    sq method.
    :return: returns an array of feature names with the 16 best scores
    """
    
    if os.path.isfile('Zasd.csv') != True:
        raise Exception("File not exist")

        
        # read data from csv file
    dataframe = pd.read_csv('Zasd.csv')
    
    if dataframe.empty:
        raise Exception('CSV file is empty')

    
    datfile = open('Zasd.csv','r')
    dat = [line.split(',') for line in datafile.readlines()]
    
    for i in dat[1:]:
        if len(i) > len(dat[0]):
            raise Exception("There is data row that is longer then the header row.")
            
    for i in dat[0]:
        if blank.match(i).end() == len(i):
            raise Exception("Empty header column found")

        
    
    # drop target columns
    drop_cols = ['Cath']
    X = dataframe.drop(drop_cols, axis=1)  # define the independent columns
    y = dataframe['Cath']  # set y as the target column for the model (what should be predicted)
    
    # initialize SelectKBest to dind 16 best features
    best_features = SelectKBest(chi2, k=16)
    fit = best_features.fit(X, y)
    df_scores = pd.DataFrame(fit.scores_) # fit dataframe into list 
    df_columns = pd.DataFrame(X.columns)

    feature_scores = pd.concat([df_columns, df_scores], axis=1)
    feature_scores.columns = ['Feature_Name', 'Score']  
    
        # export the selected features to a .csv file
    df_univ_feat = feature_scores.nlargest(16, 'Score')
    df_univ_feat.to_csv('feature_selection_UNIVARIATE.csv', index=False)
    
    return (feature_scores.nlargest(16,'Score'))





    

def treeClassifier():
    
    """
    Funtion calls feature selection methods in the using a decision tree module to determine the best features using the tree classifier method.
    :return: returns an array of feature names with the 16 best scores
    """
        
        
    if os.path.isfile('Zasd.csv') != True:
        raise Exception("File not exist")

        
        # read data from csv file
    dataframe = pd.read_csv('Zasd.csv')
    
    if dataframe.empty:
        raise Exception('CSV file is empty')

    
    datfile = open('Zasd.csv','r')
    dat = [line.split(',') for line in datafile.readlines()]
    
    for i in dat[1:]:
        if len(i) > len(dat[0]):
            raise Exception("There is data row that is longer then the header row.")
            
    for i in dat[0]:
        if blank.match(i).end() == len(i):
            raise Exception("Empty header column found")
        
        
    dataframe = pd.read_csv("Zasd.csv")
    
    X = dataframe.iloc[:,0:54]  #set independent columns
    y = dataframe.iloc[:,-1]    #set target column in this case- 'Cath'
    model = ExtraTreesClassifier()
    model.fit(X,y)
  
    #plot graph of features 
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(16).plot(kind='barh')
    
    # plt.show()
    # print(feat_importances.nlargest(20))
    return (feat_importances.nlargest(16))

# chiSquared()
# treeClassifier()
