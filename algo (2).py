import pandas as pd
import statistics
import matplotlib.pyplot as plt
import numpy as np
import sklearn   #scipy install
from sklearn.ensemble import GradientBoostingClassifier #For Classification
from sklearn.ensemble import GradientBoostingRegressor #For Regression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB



def algo(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal):
    df = (pd.DataFrame(pd.read_csv('C:\\Users\\siva\\Desktop\\hearttrain.csv')))
    ef = (pd.DataFrame(pd.read_csv('C:\\Users\\siva\\Desktop\\hearttrain.csv')))
    X =df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
    Y = df[['target']]
    A =df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
    B = df[['target']]
    a_train, a_test, b_train, b_test = train_test_split(A, B, test_size=0.3)


    if age == "" or sex == "" or cp == "" or trestbps== "" or chol== "" or fbs == "" or restecg == "" or thalach == "" or exang == "" or exang == "" or oldpeak == "" or slope == "" or ca == "" or thal == "" :
        return "Missing values"
    
    '''
    #gradientboostingclassifier
    dlf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
    dlf.fit(a_train,b_train)
    b_pred=dlf.predict(a_test)
    gba=metrics.accuracy_score(b_test, b_pred)
    print('gradientboostclassifier: ',gba)
    print(sklearn.metrics.confusion_matrix(b_test,b_pred))
    


    '''
    #decision tree
    model = tree.DecisionTreeClassifier(criterion='gini')  # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini
    model.fit(a_train, b_train)
    b_pred = model.predict(a_test)
    gba = metrics.accuracy_score(b_test, b_pred)
    print("Decision tree:",gba)
    print(sklearn.metrics.confusion_matrix(b_test, b_pred))
    '''
    #svm
    sv=svm.SVC(gamma='scale')
    sv.fit(a_train, b_train)
    b_pred = sv.predict(a_test)
    gba = metrics.accuracy_score(b_test, b_pred)
    #print(sklearn.metrics.confusion_matrix(b_test, b_pred))
    print("Svm:", gba)
    
    #naive bayes
    nb=GaussianNB()
    nb.fit(a_train, b_train)
    y_pred = nb.predict(a_test)
    gba = metrics.accuracy_score(b_test, y_pred)
    print("Naive Bayes:", gba)




    #random forest classifier
    rf=RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None,min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False,class_weight=None)
    rf.fit(a_train, b_train)
    b_pred = rf.predict(a_test)
    gba = metrics.accuracy_score(b_test, b_pred)
    print("RFC", gba)
    #print(sklearn.metrics.confusion_matrix(b_test, b_pred))
    
    

   
    '''

    gbp = model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    print(gbp)

    if gbp == 0:
        result = "Yes"
    else:
        result = "No"
    return result


