#data preprocessing

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.externals import joblib

def train_model():
    #importing dataset
    dataset =pd.read_csv(r"C:\Users\priya\Project\model\PcodMini.csv")
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, 19].values

    #encoding categorical data
    labelencoder_y=LabelEncoder()
    Y=labelencoder_y.fit_transform(Y)

    #spliting training and test set
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


    #fitting the classifier to the training set
    classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
    classifier.fit(X_train, Y_train)

    #predicting the test set results
    Y_pred=classifier.predict(X_test)

    # making the confusion matrix and calculating accuraccy
    cm=confusion_matrix(Y_test,Y_pred)
    accuracy = accuracy_score(Y_test, Y_pred)

    #dumping the classifier into model
    joblib.dump(classifier, 'pcod-model.model')
    print('Model Training Finished.\n\tAccuracy obtained: {}'.format(accuracy))