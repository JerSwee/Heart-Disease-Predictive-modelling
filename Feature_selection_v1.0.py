import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_regression
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

def chiSquared():
    '''
    dataframe = pd.read_csv("Zasd.csv")
    array = dataframe.values
    X = array[:, 0:55]
    Y = array[:, 55]
    test = SelectKBest(score_func=chi2, k=25)
    fit = test.fit(X, Y)

    # Summarize scores
    np.set_printoptions(precision=3)
    print(fit.scores_)

    features = fit.transform(X)
    # Summarize selected features
    print(features[0:26,:])
     '''



    data_clf = pd.read_csv('Zasd.csv')  # for classification problem

    print(data_clf.head(2))

    X_clf = data_clf.iloc[:, 0:55]
    y_clf = data_clf.iloc[:, 55]

    X_clf_new = SelectKBest(score_func=chi2, k=25).fit_transform(X_clf, y_clf)
    print(X_clf_new[:55])
    print(X_clf.head())


def treeClassifier():
    data = pd.read_csv("Zasd.csv")
    X = data.iloc[:,0:55]  #independent columns
    y = data.iloc[:,-1]    #target column i.e price range
    model = ExtraTreesClassifier()
    model.fit(X,y)
    print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(25).plot(kind='barh')
    plt.show()
    print(feat_importances.nlargest(25))


chiSquared()
treeClassifier()
