import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


pd.set_option('display.max_columns', None)

#Function to load the csv data into the pandas DataFrame
def load_data():
    return pd.read_csv("Datasets/drug200.csv")


#plotting the classes
#print(df)
#data.plot(kind = 'scatter', x = 'Sex', y = 'Drug', color = 'red')
#plt.show() 


#Using pandas.get_dummies to convert the ordinal and nomnal features in numerical format
df = load_data()
df = pd.get_dummies(df, columns=['Sex', 'BP', 'Cholesterol'])
#print(df.head())


#Takes all of the data from the Drug column
y = df.Drug

#Takes the data of all the csv file, excluding the Drug column
x = df.drop('Drug', axis=1)

#Splitting the data into training and test set
#25% of the data is going into test
#75% is of the data going into trainning
x_train, x_test, y_train, y_test = train_test_split(x,y)

print(y_test)

#a) Using Gaussian Naive Bayes classifier to predict the split dataset
#   and will print out the predicted outcome of the Naive Bayes
#   classifier
classifierNB = GaussianNB()
classifierNB.fit(x_train, y_train)

predictionNB = classifierNB.predict(x_test)
print(predictionNB)


#b) using Decision Tree classifier to predict the split dataset
#   and will print out the predicted outcome of the Base-DT
#   classifier
classifierBase_DT = DecisionTreeClassifier()
classifierBase_DT.fit(x_train, y_train)

predictionBase_DT = classifierBase_DT.predict(x_test)
print(predictionBase_DT)


#d) Using Perceptron classifier to predict the split dataset
#   and will print out the predicted outcome of the perceptron
#   classifier
classifierPercep = Perceptron()
classifierPercep.fit(x_train, y_train)

predictionPercep = classifierPercep.predict(x_test)
print(predictionPercep)


#e) Using Base-MLP classifier to predict the split dataset
#   and will print out the predicted outcome of the Base-MLP
#   classifier
classifierBase_MLP = MLPClassifier(activation='logistic', solver='sgd')
classifierBase_MLP.fit(x_train, y_train)

preditionBase_MLP = classifierBase_MLP.predict(x_test)
print(preditionBase_MLP)
