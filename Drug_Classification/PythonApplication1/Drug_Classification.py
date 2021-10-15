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
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

pd.set_option('display.max_columns', None)

#Function to load the csv data into the pandas DataFrame
def load_data():
    return pd.read_csv("Datasets/drug200.csv")

#Loading the dataFrame into the variable df
df = load_data()

#storing the drug column into the variable drugCol
drugCol = df.iloc[:,-1:].values
drugList = []

#Appending unique drug names into a new list and sorting it 
#in alphabetical order
for drug in drugCol:
    if drug[0] not in drugList:
        drugList.append(drug[0])

sortedDrugList = sorted(drugList)

drugFrequency = [0] * len(sortedDrugList)

#For loop that will count the number of every drug type
for drug in drugCol:
    index = sortedDrugList.index(drug[0])
    drugFrequency[index] = drugFrequency[index] + 1

#Plotting the frequency of each drug type
plt.figure(figsize=(9, 5))
plt.xlabel('Drugs')
plt.ylabel('Instances')
plt.bar(sortedDrugList, drugFrequency)
plt.suptitle('Frequency of Drugs')
plt.savefig("drug-distribution.pdf")


#Using pandas.get_dummies to convert the ordinal and nomnal features in numerical format
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
print("Naive Bayes Prediction Results: ")
print(predictionNB)


#b) using Decision Tree classifier to predict the split dataset
#   and will print out the predicted outcome of the Base-DT
#   classifier
classifierBase_DT = DecisionTreeClassifier()
classifierBase_DT.fit(x_train, y_train)

predictionBase_DT = classifierBase_DT.predict(x_test)
print("Decision Prediction Results: ")
print(predictionBase_DT)

#c) Using Top-DT classifier to predict the split dataset
#   and will print out the predicted outcome of the Top-DT
#  classifier
param_TopDT = {'criterion':['entropy'], 
               'max_depth': [5, 20], 
               'min_samples_split':[3, 6, 9]}

classifierTop_DT = GridSearchCV(DecisionTreeClassifier(), param_TopDT)
classifierTop_DT.fit(x_train, y_train)

predictionTop_DT = classifierTop_DT.predict(x_test)
print("Top-DT Results: ")
print(predictionTop_DT)


#d) Using Perceptron classifier to predict the split dataset
#   and will print out the predicted outcome of the perceptron
#   classifier
classifierPercep = Perceptron()
classifierPercep.fit(x_train, y_train)

predictionPercep = classifierPercep.predict(x_test)
print("Perceptron Prediction Results: ")
print(predictionPercep)


#e) Using Base-MLP classifier to predict the split dataset
#   and will print out the predicted outcome of the Base-MLP
#   classifier
classifierBase_MLP = MLPClassifier(activation='logistic', solver='sgd')
classifierBase_MLP.fit(x_train, y_train)

predictionBase_MLP = classifierBase_MLP.predict(x_test)
print("MLP Prediction Results: ")
print(predictionBase_MLP)


#f) Using Top-MLP classifier to predict the split dataset
#   and will print out the predicted outcome of the Top-MLP
#   classifier
param_TopMLP = {'activation': ['tanh', 'relu', 'identity'],
               'hidden_layer_sizes': [(30, 50, 150), (8, 24, 116)],
               'solver': ['adam', 'sgd']        
               }

classifierTop_MLP = GridSearchCV(estimator = MLPClassifier(max_iter=200), param_grid = param_TopMLP, n_jobs=-1)
classifierTop_MLP.fit(x_train, y_train)

predictionTop_MLP = classifierTop_MLP.predict(x_test)
print("Top-MLP Prediction Results: ")
print(predictionTop_MLP)