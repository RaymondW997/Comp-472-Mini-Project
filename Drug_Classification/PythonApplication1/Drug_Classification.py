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
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

pd.set_option('display.max_columns', None)

#Function to load the csv data into the pandas DataFrame
def load_data():
    return pd.read_csv("Datasets/drug200.csv")

if __name__ == '__main__':
    #Loading the dataFrame into the variable df
    df = load_data()

    expected = df['Drug'].values
    # print(expected)

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

    print("expected test values")
    # print(y_test)
    y_test_array = y_test.to_numpy()
    print(y_test_array)

    #a) Using Gaussian Naive Bayes classifier to predict the split dataset
    #   and will print out the predicted outcome of the Naive Bayes
    #   classifier
    print("======================================================================")
    classifierNB = GaussianNB()
    classifierNB.fit(x_train, y_train)

    predictionNB = classifierNB.predict(x_test)
    print("Naive Bayes Prediction Results: ")
    print(predictionNB)
    print("Confusion matrix for Gaussian NB: ")
    matrixNB = confusion_matrix(y_test_array, predictionNB)
    print(matrixNB)
    print("Precision, Recall, and F1-Score per drug")
    precisionNB, recallNB, f1NB, supportNB = precision_recall_fscore_support(y_test_array, predictionNB, average=None, labels=["drugA", "drugB", "drugC", "drugX", "drugY"])
    print("\t\t\t\tdrugA\t\tdrugB\t\tdrugC\t\tdrugX\t\tdrugY")
    print("Precision\t", precisionNB)
    print("Recall\t\t", recallNB)
    print("F1-Score\t", f1NB)
    print("Occurences\t", supportNB)
    print()
    precisionNBmac, recallNBmac, f1NBmac, supportNBmac = precision_recall_fscore_support(y_test_array, predictionNB, average='macro')
    precisionNBwtd, recallNBwtd, f1NBwtd, supportNBwtd = precision_recall_fscore_support(y_test_array, predictionNB, average='weighted')
    accuracyNB = accuracy_score(y_test_array, predictionNB)
    print("Accuracy\t\t\t", accuracyNB)
    print("Macro-Average F1\t", f1NBmac)
    print("Weighted-Average F1\t", f1NBwtd)

    #b) using Decision Tree classifier to predict the split dataset
    #   and will print out the predicted outcome of the Base-DT
    #   classifier
    print("======================================================================")
    classifierBase_DT = DecisionTreeClassifier()
    classifierBase_DT.fit(x_train, y_train)

    predictionBase_DT = classifierBase_DT.predict(x_test)
    print("Decision Prediction Results: ")
    print(predictionBase_DT)
    print("Confusion matrix for base DT: ")
    matrixBDT = confusion_matrix(y_test_array, predictionBase_DT, labels=["drugA", "drugB", "drugC", "drugX", "drugY"])
    print(matrixBDT)
    precisionBDT, recallBDT, f1BDT, supportBDT = precision_recall_fscore_support(y_test_array, predictionBase_DT, average=None,
                                                                             labels=["drugA", "drugB", "drugC", "drugX",
                                                                                     "drugY"])
    print("Precision, Recall, and F1-Score per drug")
    print("\t\t\t\tdrugA\t\tdrugB\t\tdrugC\t\tdrugX\t\tdrugY")
    print("Precision\t", precisionBDT)
    print("Recall\t\t", recallBDT)
    print("F1-Score\t", f1BDT)
    print("Occurences\t", supportBDT)
    print()
    precisionBDTmac, recallBDTmac, f1BDTmac, supportBDTmac = precision_recall_fscore_support(y_test_array, predictionBase_DT, average='macro')
    precisionBDTwtd, recallBDTwtd, f1BDTwtd, supportBDTwtd = precision_recall_fscore_support(y_test_array, predictionBase_DT, average='weighted')
    accuracyBDT = accuracy_score(y_test_array, predictionBase_DT)
    print("Accuracy\t\t\t", accuracyBDT)
    print("Macro-Average F1\t", f1BDTmac)
    print("Weighted-Average F1\t", f1BDTwtd)

    #c) Using Top-DT classifier to predict the split dataset
    #   and will print out the predicted outcome of the Top-DT
    #  classifier
    print("======================================================================")
    param_TopDT = {'criterion': ['gini', 'entropy'],
                   'max_depth': [5, 20],
                   'min_samples_split':[3, 6, 9]}

    classifierTop_DT = GridSearchCV(DecisionTreeClassifier(), param_TopDT, scoring='accuracy')
    classifierTop_DT.fit(x_train, y_train)
    print("Best hyperparameters as selected by GridSearchCV: ")
    print(classifierTop_DT.best_params_)

    predictionTop_DT = classifierTop_DT.predict(x_test)
    print("Top-DT Results: ")
    print(predictionTop_DT)
    print("Confusion matrix for top DT: ")
    matrixTDT = confusion_matrix(y_test_array, predictionTop_DT, labels=["drugA", "drugB", "drugC", "drugX", "drugY"])
    print(matrixTDT)
    print("Precision, Recall, and F1-Score per drug")
    precisionTDT, recallTDT, f1TDT, supportTDT = precision_recall_fscore_support(y_test_array, predictionTop_DT, average=None,
                                                                             labels=["drugA", "drugB", "drugC", "drugX",
                                                                                     "drugY"])
    print("\t\t\t\tdrugA\t\tdrugB\t\tdrugC\t\tdrugX\t\tdrugY")
    print("Precision\t", precisionTDT)
    print("Recall\t\t", recallTDT)
    print("F1-Score\t", f1TDT)
    print("Occurences\t", supportTDT)
    print()
    precisionTDTmac, recallTDTmac, f1TDTmac, supportTDTmac = precision_recall_fscore_support(y_test_array,
                                                                                             predictionTop_DT,
                                                                                             average='macro')
    precisionTDTwtd, recallTDTwtd, f1TDTwtd, supportTDTwtd = precision_recall_fscore_support(y_test_array,
                                                                                             predictionTop_DT,
                                                                                             average='weighted')
    accuracyTDT = accuracy_score(y_test_array, predictionTop_DT)
    print("Accuracy\t\t\t", accuracyTDT)
    print("Macro-Average F1\t", f1TDTmac)
    print("Weighted-Average F1\t", f1TDTwtd)

    #d) Using Perceptron classifier to predict the split dataset
    #   and will print out the predicted outcome of the perceptron
    #   classifier
    print("======================================================================")
    classifierPercep = Perceptron()
    classifierPercep.fit(x_train, y_train)

    predictionPercep = classifierPercep.predict(x_test)
    print("Perceptron Prediction Results: ")
    print(predictionPercep)
    print("Confusion matrix for perceptron: ")
    matrixPER = confusion_matrix(y_test_array, predictionPercep, labels=["drugA", "drugB", "drugC", "drugX", "drugY"])
    print(matrixPER)
    print("Precision, Recall, and F1-Score per drug")
    precisionPER, recallPER, f1PER, supportPER = precision_recall_fscore_support(y_test_array, predictionPercep, average=None,
                                                                             labels=["drugA", "drugB", "drugC", "drugX",
                                                                                     "drugY"])
    print("\t\t\t\tdrugA\t\tdrugB\t\tdrugC\t\tdrugX\t\tdrugY")
    print("Precision\t", precisionPER)
    print("Recall\t\t", recallPER)
    print("F1-Score\t", f1PER)
    print("Occurences\t", supportPER)
    print()
    precisionPERmac, recallPERmac, f1PERmac, supportPERmac = precision_recall_fscore_support(y_test_array,
                                                                                             predictionPercep,
                                                                                             average='macro')
    precisionPERwtd, recallPERwtd, f1PERwtd, supportPERwtd = precision_recall_fscore_support(y_test_array,
                                                                                             predictionPercep,
                                                                                             average='weighted')
    accuracyPER = accuracy_score(y_test_array, predictionPercep)
    print("Accuracy\t\t\t", accuracyPER)
    print("Macro-Average F1\t", f1PERmac)
    print("Weighted-Average F1\t", f1PERwtd)

    #e) Using Base-MLP classifier to predict the split dataset
    #   and will print out the predicted outcome of the Base-MLP
    #   classifier
    print("======================================================================")
    classifierBase_MLP = MLPClassifier(activation='logistic', solver='sgd')
    classifierBase_MLP.fit(x_train, y_train)

    predictionBase_MLP = classifierBase_MLP.predict(x_test)
    print("MLP Prediction Results: ")
    print(predictionBase_MLP)
    print("Confusion matrix for base MLP: ")
    matrixBMLP = confusion_matrix(y_test, predictionBase_MLP, labels=["drugA", "drugB", "drugC", "drugX", "drugY"])
    print(matrixBMLP)
    print("Precision, Recall, and F1-Score per drug")
    precisionBMLP, recallBMLP, f1BMLP, supportBMLP = precision_recall_fscore_support(y_test_array, predictionBase_MLP, average=None,
                                                                             labels=["drugA", "drugB", "drugC", "drugX",
                                                                                     "drugY"])
    print("\t\t\t\tdrugA\t\tdrugB\t\tdrugC\t\tdrugX\t\tdrugY")
    print("Precision\t", precisionBMLP)
    print("Recall\t\t", recallBMLP)
    print("F1-Score\t", f1BMLP)
    print("Occurences\t", supportBMLP)
    print()
    precisionBMLPmac, recallBMLPmac, f1BMLPmac, supportBMLPmac = precision_recall_fscore_support(y_test_array,
                                                                                             predictionBase_MLP,
                                                                                             average='macro')
    precisionBMLPwtd, recallBMLPwtd, f1BMLPwtd, supportBMLPwtd = precision_recall_fscore_support(y_test_array,
                                                                                             predictionBase_MLP,
                                                                                             average='weighted')
    accuracyBMLP = accuracy_score(y_test_array, predictionBase_MLP)
    print("Accuracy\t\t\t", accuracyBMLP)
    print("Macro-Average F1\t", f1BMLPmac)
    print("Weighted-Average F1\t", f1BMLPwtd)

    #f) Using Top-MLP classifier to predict the split dataset
    #   and will print out the predicted outcome of the Top-MLP
    #   classifier
    print("======================================================================")
    param_TopMLP = {'activation': ['tanh', 'relu', 'identity'],
                    'hidden_layer_sizes': [(30, 50, 150), (8, 24, 116)],
                    'solver': ['adam', 'sgd']
                    }

    classifierTop_MLP = GridSearchCV(estimator = MLPClassifier(max_iter=200), param_grid = param_TopMLP, n_jobs=-1, scoring='accuracy')
    classifierTop_MLP.fit(x_train, y_train)
    print("Best hyperparameters as selected by GridSearchCV: ")
    print(classifierTop_DT.best_params_)

    predictionTop_MLP = classifierTop_MLP.predict(x_test)
    print("Top-MLP Prediction Results: ")
    print(predictionTop_MLP)
    print("Confusion matrix for top MLP: ")
    matrixTMLP = confusion_matrix(y_test_array, predictionTop_MLP, labels=["drugA", "drugB", "drugC", "drugX", "drugY"])
    print(matrixTMLP)
    print("Precision, Recall, and F1-Score per drug")
    precisionTMLP, recallTMLP, f1TMLP, supportTMLP = precision_recall_fscore_support(y_test_array, predictionTop_MLP, average=None,
                                                                             labels=["drugA", "drugB", "drugC", "drugX",
                                                                                     "drugY"])
    print("\t\t\t\tdrugA\t\tdrugB\t\tdrugC\t\tdrugX\t\tdrugY")
    print("Precision\t", precisionTMLP)
    print("Recall\t\t", recallTMLP)
    print("F1-Score\t", f1TMLP)
    print("Occurences\t", supportTMLP)
    print()
    precisionTMLPmac, recallTMLPmac, f1TMLPmac, supportTMLPmac = precision_recall_fscore_support(y_test_array,
                                                                                                 predictionTop_MLP,
                                                                                                 average='macro')
    precisionTMLPwtd, recallTMLPwtd, f1TMLPwtd, supportTMLPwtd = precision_recall_fscore_support(y_test_array,
                                                                                                 predictionTop_MLP,
                                                                                                 average='weighted')
    accuracyTMLP = accuracy_score(y_test_array, predictionTop_MLP)
    print("Accuracy\t\t\t", accuracyTMLP)
    print("Macro-Average F1\t", f1TMLPmac)
    print("Weighted-Average F1\t", f1TMLPwtd)