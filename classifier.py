# Packages for data analysis
import Feature_Selection_v1 as FS

# Packages for Classifiers
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier

# Packages for Cross Validation
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import pandas as pd


############################################################################################
#                              FEATURE SELECTION VALUES                                    #
############################################################################################

def chiSqFeatures():
    """
    Funtion calls feature selection methods in the Feature selection module to determine the best features using the chi
    sq method.
    :return: returns an array of feature name strings
    """
    chiSqFeatureResults = FS.chiSquared()
    chiSqFeatures = []
    for key in chiSqFeatureResults['Feature_Name']:
        chiSqFeatures.append(str(key))
    return chiSqFeatures


def TreeFeatures():
    """
    Funtion calls feature selection methods in the Feature selection module to determine the best features using the tree classifier
    method
    :return: returns an array of feature name strings
    """
    treeFeatureResults = FS.treeClassifier()
    treeFeatures = []
    for key in treeFeatureResults.keys():
        treeFeatures.append(str(key))
    return treeFeatures


def initialization():
    """
    Function that Initializes the program by returning the dataset values corresponding to the selected feature values and their corresponding results. Due to the hardcoded nature of the GUI,
    the list of feature values are hardcoded in, values were determined using the Treefeatures and chisqFeatures, with tree features being selected for the program
    :return: data set values corresponding to both the selected features and their corresponding result values
    """
    dataset = pd.read_csv('Zasd.csv')  # Read DataSet from csv file
    type_label = dataset['Cath'].values  # Obtain type label for features
    treeFeatures = ['Typical Chest Pain','Atypical','HTN','Age','Region RWMA','Nonanginal','Tinversion','DM','EF-TTE','Weight','BMI','TG','ESR','BP','Neut']  # feature values based on Tree feature selection method
    featureValues = dataset[treeFeatures].values  # obtain feature values based on tree features
    return featureValues,type_label


featureValues, type_label = initialization() # setting up values for use with other methods/functions

############################################################################################
#                                     CROSS VALIDATION                                     #
############################################################################################

def resultValidation(model_classifier,featureValues,type_label,validator = 1):
    """
    Function estimates the models prediction accuracy by comparing the predictive results from 4 different cross validation scores and returning the value
    Used for evaluation, by default returns a Kfold cross validator with n_splits = 10
    :param model_classifier: Input Classifier
    :param featureValues: List of Feature values
    :param type_label: List of label results
    :return: a string of the model's prediction accuracy
    """
    # K-FOLD CROSS VALIDATION
    if (validator==1):
        kfold = KFold(n_splits=10)  # <- Change split number here
        model_Kfold = model_classifier
        results_Kfold = model_selection.cross_val_score(model_Kfold,featureValues,type_label,cv=kfold)
        return '{0:.2f}'.format(results_Kfold.mean()*100.0)

    # STRATIFIED K-FOLD CROSS VALIDATION
    elif (validator==2):
        skfold = StratifiedKFold(n_splits= 10) # <- Change split number here
        model_SKfold = model_classifier
        results_SKfold = model_selection.cross_val_score(model_SKfold,featureValues,type_label,cv=skfold)
        return '{0:.2f}'.format(results_SKfold.mean()*100.0)

    # LEAVE ONE OUT CROSS VALIDATION (LOOCV)
    elif (validator==3):
        loocv = model_selection.LeaveOneOut()
        model_loocv = model_classifier
        results_loocv = model_selection.cross_val_score(model_loocv,featureValues,type_label,cv=loocv)
        return '{0:.2f}'.format(results_loocv.mean()*100.0)

    # REPEATED RANDOM TEST-TRAIN SPLITS
    else:
        rrtt = model_selection.ShuffleSplit(n_splits=10, test_size=0.30, random_state=100)  # <- Change split number and test_size here
        model_shufflecv = model_classifier
        results_4 = model_selection.cross_val_score(model_shufflecv, featureValues, type_label, cv=rrtt)
        return '{0:.2f}'.format(results_4.mean()*100.0)


############################################################################################
#                                     Parameter Tuning                                     #
############################################################################################



def kNNTuning():
    """
    Function used to evaluate the k parameter for kNN, iterates through 1-20 neighbours and prints results
    :return: The predictive accuracy using the input number of neighbours
    """
    for k in range(1,21):
        baseclassifer = KNeighborsClassifier(k)
        KnnResults = resultValidation(baseclassifer, featureValues, type_label)
        print(KnnResults)


def SvmTuner():
    """
    Function used to evaluate the parameters for the SVM classifier, evaluates different Kernals, C and gamma values to evaluate the combination that yields the best results
    :return: string output evaluation of test results
    """
    X_train, X_test, y_train, y_test = train_test_split(featureValues, type_label, test_size=0.3,random_state=100)  # <- Change test sizes here
    param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
    grid.fit(X_train, y_train)
    print(grid.best_estimator_)



############################################################################################
#                               ENSEMBLE METHOD VARIETIES                                  #
############################################################################################

###########
# BAGGING #
###########

# BAGGING CLASSIFICATION
def testbagging():
    """
    Function used to evaluate the best base classifier used to for bagging, classifiers tested - kNN with 7 neighbours, naive bayes, decision tree. tested over a range of 200 estimators
    :return: string result of predictive accuracy
    """
    baseClassifiers = []
    model_knn = KNeighborsClassifier(7)
    baseClassifiers.append(('KNN', model_knn))
    model_nb = GaussianNB()
    baseClassifiers.append(('NB', model_nb))
    model_dt = DecisionTreeClassifier()
    baseClassifiers.append(('DT', model_dt))
    # model_svm = SVC(kernel='poly', C=0.1, gamma=0.001) # Exclude/Run SVM individually, as it has a much longer run time compared to other classifiers when paired with a bagging ensemble
    # baseClassifiers.append(('SVM', model_svm))
    for classifier in baseClassifiers:
        print("classifier: "+classifier[0])
        print("================================================================================")
        for x in range(1, 201):
            baggingModel = BaggingClassifier(base_estimator=classifier[1],n_estimators=x,random_state=10)
            print(resultValidation(baggingModel, featureValues, type_label))


def bagging(x=100):
    """
    Returns bagging ensemble classifier using NaiveBayes classifier as the base naive bayes was chosen when compared to other classifiers, by default sets the number to 100 which is the number determined via classifier evaluation
    :return: bagging Ensemble classifier with NaiveBayes as base with x estimators
    """
    model_nb = GaussianNB()
    baggingNB = BaggingClassifier(base_estimator=model_nb, n_estimators=x, random_state=10) # <- Ensemble using Decision tree as Base classifier
    return baggingNB


# RANDOM FOREST CLASSIFICATION
def randomForest(x=100):
    """
    Function returns a random forest ensemble Classifier with input number of estimators, by default sets the number to 100 which is the number determined via classifier evaluation,
    :param x: number of estimators
    :return: adaBoost classifier with x estimators
    """
    model_rf = RandomForestClassifier(n_estimators=x)
    return model_rf

def testRF():
    """
    Function used to test the random Forest classifier with varying number of estimators ranging (1-200) and prints out the cross validation result
    :return: string result of percentage accuracy
    """
    for x in range(1,201):
        print(resultValidation(randomForest(x),featureValues,type_label))

############
# Boosting #
############

# ADABOOST CLASSIFIER ENSEMBLE

def adaBoost(x = 30):
    """
    Function returns a adaBoost Classifier with input number of estimators, by default sets the number to 30 which is the number determined via classifier evaluation,
    :param x: number of estimators
    :return: adaBoost classifier with x estimators
    """
    model_ada = AdaBoostClassifier(n_estimators=x, random_state=10)
    return model_ada

def testAda():
    """
    Function used to test the ada boost classifier with varying number of estimators ranging (1-200) and prints out the cross validation result
    :return: string result of percentage accuracy
    """
    for x in range(1,201):
        print(resultValidation(adaBoost(x),featureValues,type_label))

# STOCHASTIC GRADIENT BOOSTING
def sgb(x=100):
    """
    Function returns a SGB Classifier with input number of estimators, by default sets the number to 100 which is the number determined via classifier evaluation,
    :param x: number of estimators
    :return: sgb classifier with x estimators
    """
    model_sgb = GradientBoostingClassifier(n_estimators=x, random_state=10)
    return model_sgb

def testSgb():
    """
    Function used to test the sgb classifier with varying number of estimators ranging (1-200) and prints out the cross validation result
    :return:  string result of percentage accuracy
    """
    for x in range(1, 201):
        print(resultValidation(adaBoost(x), featureValues, type_label))


# VOTING ENSEMBLE
def voting():
    """
    Function returns a voting ensemble containing one of each of the following classifiers LDA,KNN,NB,DT and SVM, with each parameters set to values
    tested to yield the better predictive accuracy prior
    :return:  classifier with the afore mentioned classifiers
    """
    baseClassifiers = []
    # Create base classifier and add to list, change type and number of classifiers to add here
    model_lda = LinearDiscriminantAnalysis()
    baseClassifiers.append(('LDA',model_lda))
    model_knn = KNeighborsClassifier(7)
    baseClassifiers.append(('KNN',model_knn))
    model_nb = GaussianNB()
    baseClassifiers.append(('NB',model_nb))
    # model_dt = DecisionTreeClassifier()
    # baseClassifiers.append(('DT',model_dt))
    model_svm = SVC(kernel='poly',C=0.1,gamma=0.001)
    baseClassifiers.append(('SVM',model_svm))
    votingEnsemble = VotingClassifier(baseClassifiers)
    return votingEnsemble

############################################################################################
#                                HTML PROGRAM FUNCTIONS                                    #
############################################################################################


def getClassfier(classiferName):
    """
    Simple selector to return a classifier based on string input, used to link the HTML selectors used for the project back to python code
    :param classiferName: string name of classifier
    :return: classifier of the requested type
    """
    if classiferName == "voting":
        return voting()
    elif classiferName == "sgb":
        return sgb()
    elif classiferName == "ada":
        return adaBoost()
    elif classiferName == "rf":
        return randomForest()
    elif classiferName == "bag":
        return bagging()


def predictResult(TypicalChestPain,Atypical,HTN,Age,RWMA,Nonanginal,Tinversion,DM,EF,Weight,BMI,TG,ESR,BP,Neut,classifierName):
    """
    Main Predictor Function used by the program, takes in 15 feature values for model prediction determined by feature selection in addition to the classifer name, values are as such
    :param TypicalChestPain: 0/1 value representing presence of absence of typical chest pain
    :param Atypical: 0/1 value representing presence of absence of Atypical chest pain
    :param HTN: 0/1 value representing if the patient suffers from hypertension
    :param Age: integer value representing patients age
    :param RWMA: Value ranging 0-4 representing the severity of Regional Wall Motion Abnormalities suffered by patient
    :param Nonanginal: 0/1 value representing if the patient suffers from Nonanginal chest pain
    :param Tinversion: 0/1 value representing if the patient has a T-wave inversion
    :param DM: 0/1 value representing if the patient suffers form diabetes
    :param EF: 0/100 percentage value of Ejection Fraction
    :param Weight: Weight of patient
    :param BMI: BMI of patient
    :param TG: Triglyceride value of patient
    :param ESR: Erythrocyte Sedimentation rate
    :param BP: Blood Pressure of patient
    :param Neut: Patient Neutrophil Percentage
    :param classifierName: Name of classifier to call
    :return: returns the prediction of the model, a value of 0 representing a normal diagnosis and a value of 1 representing a CAD diagnosis
    """
    model = getClassfier(classifierName)
    model.fit(featureValues,type_label)
    array = [TypicalChestPain,Atypical,HTN,Age,RWMA,Nonanginal,Tinversion,DM,EF,Weight,BMI,TG,ESR,BP,Neut]
    return model.predict([array])

def getAccuracy(classiferName):
    """
    Funtion returns the estimated accuracy of the chosen classifer
    :param classiferName: name of classifier
    :return: formatted string of calculated predictive accuracy
    """
    classifier = getClassfier(classiferName)
    kfold = KFold(n_splits=10)  # <- Change split number here
    results_Kfold = model_selection.cross_val_score(classifier, featureValues, type_label, cv=kfold)
    return '{0:.2f}'.format(results_Kfold.mean()*100.0)
