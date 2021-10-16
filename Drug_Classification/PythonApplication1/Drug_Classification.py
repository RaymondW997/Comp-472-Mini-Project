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
import statistics
from statistics import mean

pd.set_option('display.max_columns', None)


# Function to load the csv data into the pandas DataFrame
def load_data():
    return pd.read_csv("Datasets/drug200.csv")


if __name__ == '__main__':
    # Loading the dataFrame into the variable df
    df = load_data()

    # opening file to write output to
    outputFile = open("../drugs-performace.txt", 'w')

    # storing the drug column into the variable drugCol
    drugCol = df.iloc[:, -1:].values
    drugList = []

    # Appending unique drug names into a new list and sorting it
    # in alphabetical order
    for drug in drugCol:
        if drug[0] not in drugList:
            drugList.append(drug[0])

    sortedDrugList = sorted(drugList)

    drugFrequency = [0] * len(sortedDrugList)

    # For loop that will count the number of every drug type
    for drug in drugCol:
        index = sortedDrugList.index(drug[0])
        drugFrequency[index] = drugFrequency[index] + 1

    # Plotting the frequency of each drug type
    plt.figure(figsize=(9, 5))
    plt.xlabel('Drugs')
    plt.ylabel('Instances')
    plt.bar(sortedDrugList, drugFrequency)
    plt.suptitle('Frequency of Drugs')
    plt.savefig("drug-distribution.pdf")

    # Using pandas.get_dummies to convert the ordinal and nomnal features in numerical format
    df = pd.get_dummies(df, columns=['Sex', 'BP', 'Cholesterol'])
    # print(df.head())

    # Takes all of the data from the Drug column
    y = df.Drug
    # Takes the data of all the csv file, excluding the Drug column
    x = df.drop('Drug', axis=1)

    # Splitting the data into training and test set
    # 25% of the data is going into test
    # 75% is of the data going into trainning
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    print("Expected test values")
    outputFile.write("Expected test values\n")
    # print(y_test)
    y_test_array = y_test.to_numpy()
    print(y_test_array)
    outputFile.writelines(str(y_test_array))
    outputFile.write('\n')

    #creating empty arrays for averages and standard deviations
    NBAcc, NBmac, NBwtd = list(), list(), list()
    BDTAcc, BDTmac, BDTwtd = list(), list(), list()
    TDTAcc, TDTmac, TDTwtd = list(), list(), list()
    PerAcc, Permac, Perwtd = list(), list(), list()
    BMLPAcc, BMLPmac, BMLPwtd = list(), list(), list()
    TMLPAcc, TMLPmac, TMLPwtd = list(), list(), list()

    # a) Using Gaussian Naive Bayes classifier to predict the split dataset
    #   and will print out the predicted outcome of the Naive Bayes
    #   classifier
    print("======================================================================")
    outputFile.write("======================================================================\n")
    print("(a) NB: The Gaussian Naive Bayes Classifier")
    outputFile.write("(a) NB: The Gaussian Naive Bayes Classifier")
    classifierNB = GaussianNB()
    classifierNB.fit(x_train, y_train)

    predictionNB = classifierNB.predict(x_test)
    print("Naive Bayes Prediction Results: ")
    outputFile.write("Naive Bayes Prediction Results: \n")
    print(predictionNB)
    print("Confusion matrix for Gaussian NB: ")
    outputFile.writelines(str(predictionNB))
    outputFile.write('\n')
    print("(b) Confusion matrix for Gaussian NB: ")
    outputFile.write("(b) Confusion matrix for Gaussian NB: \n")
    matrixNB = confusion_matrix(y_test_array, predictionNB)
    print(matrixNB)
    print("Precision, Recall, and F1-Score per drug")
    outputFile.write(str(matrixNB))
    outputFile.write('\n')
    print("(c) Precision, Recall, and F1-Score per drug")
    outputFile.write("(c) Precision, Recall, and F1-Score per drug\n")
    precisionNB, recallNB, f1NB, supportNB = precision_recall_fscore_support(y_test_array, predictionNB, average=None,
                                                                             labels=["drugA", "drugB", "drugC", "drugX",
                                                                                     "drugY"])
    print("\t\t\t\tdrugA\t\tdrugB\t\tdrugC\t\tdrugX\t\tdrugY")
    print("\t\tdrugA\t\tdrugB\t\tdrugC\t\tdrugX\t\tdrugY")
    outputFile.write("\t\tdrugA\t\tdrugB\t\tdrugC\t\tdrugX\t\tdrugY\n")
    print("Precision\t", precisionNB)
    outputFile.write("Precision\t" + str(precisionNB) + '\n')
    print("Recall\t\t", recallNB)
    outputFile.writelines("Recall\t\t" + str(recallNB) + '\n')
    print("F1-Score\t", f1NB)
    outputFile.writelines("F1-Score\t" + str(f1NB) + '\n')
    print("Occurences\t", supportNB)
    outputFile.writelines("Occurences\t" + str(supportNB) + '\n')
    print()
    outputFile.write('\n')
    precisionNBmac, recallNBmac, f1NBmac, supportNBmac = precision_recall_fscore_support(y_test_array, predictionNB,
                                                                                         average='macro')
    precisionNBwtd, recallNBwtd, f1NBwtd, supportNBwtd = precision_recall_fscore_support(y_test_array, predictionNB,
                                                                                         average='weighted')
    accuracyNB = accuracy_score(y_test_array, predictionNB)
    print("(d) The accuracy, macro-average F1 and weighted-average F1 of the model")
    outputFile.write("(d) The accuracy, macro-average F1 and weighted-average F1 of the model\n")
    print("Accuracy\t\t", accuracyNB)
    NBAcc.append(accuracyNB)
    outputFile.writelines("Accuracy\t\t" + str(accuracyNB) + '\n')
    print("Macro-Average F1\t", f1NBmac)
    NBmac.append(f1NBmac)
    outputFile.writelines("Macro-Average F1\t" + str(f1NBmac) + '\n')
    print("Weighted-Average F1\t", f1NBwtd)
    NBwtd.append(f1NBwtd)
    outputFile.writelines("Weighted-Average F1\t" + str(f1NBwtd) + '\n')

    # b) using Decision Tree classifier to predict the split dataset
    #   and will print out the predicted outcome of the Base-DT
    #   classifier
    print("======================================================================")
    print("(a) Base-DT: the Decision Tree")
    outputFile.write("======================================================================\n")
    outputFile.write("(a) Base-DT: the Decision Tree\n")
    classifierBase_DT = DecisionTreeClassifier()
    classifierBase_DT.fit(x_train, y_train)

    predictionBase_DT = classifierBase_DT.predict(x_test)
    print("Decision Prediction Results: ")
    print(predictionBase_DT)
    print("Confusion matrix for base DT: ")
    print("(b) Confusion matrix for base DT: ")
    outputFile.writelines("Decision Prediction Results: \n")
    outputFile.writelines(str(predictionBase_DT) + "\n")
    outputFile.writelines("(b) Confusion matrix for base DT: \n")
    matrixBDT = confusion_matrix(y_test_array, predictionBase_DT, labels=["drugA", "drugB", "drugC", "drugX", "drugY"])
    print(matrixBDT)
    outputFile.writelines(str(matrixBDT) + "\n")
    precisionBDT, recallBDT, f1BDT, supportBDT = precision_recall_fscore_support(y_test_array, predictionBase_DT,
                                                                                 average=None,
                                                                                 labels=["drugA", "drugB", "drugC",
                                                                                         "drugX",
                                                                                         "drugY"])
    print("Precision, Recall, and F1-Score per drug")
    print("\t\t\t\tdrugA\t\tdrugB\t\tdrugC\t\tdrugX\t\tdrugY")
    print("(c) Precision, Recall, and F1-Score per drug")
    print("\t\tdrugA\t\tdrugB\t\tdrugC\t\tdrugX\t\tdrugY")
    print("Precision\t", precisionBDT)
    print("Recall\t\t", recallBDT)
    print("F1-Score\t", f1BDT)
    print("Occurences\t", supportBDT)
    print()
    outputFile.writelines("(c) Precision, Recall, and F1-Score per drug \n")
    outputFile.writelines("\t\tdrugA\t\tdrugB\t\tdrugC\t\tdrugX\t\tdrugY\n")
    outputFile.writelines("Precision\t" + str(precisionBDT) + '\n')
    outputFile.writelines("Recall\t\t" + str(recallBDT) + '\n')
    outputFile.writelines("F1-Score\t" + str(f1BDT) + '\n')
    outputFile.writelines("Occurences\t" + str(supportBDT) + '\n')
    outputFile.writelines('\n')
    precisionBDTmac, recallBDTmac, f1BDTmac, supportBDTmac = precision_recall_fscore_support(y_test_array,
                                                                                             predictionBase_DT,
                                                                                             average='macro')
    precisionBDTwtd, recallBDTwtd, f1BDTwtd, supportBDTwtd = precision_recall_fscore_support(y_test_array,
                                                                                             predictionBase_DT,
                                                                                             average='weighted')
    accuracyBDT = accuracy_score(y_test_array, predictionBase_DT)
    print("(d) The accuracy, macro-average F1 and weighted-average F1 of the model")
    outputFile.write("(d) The accuracy, macro-average F1 and weighted-average F1 of the model\n")
    print("Accuracy\t\t", accuracyBDT)
    print("Macro-Average F1\t", f1BDTmac)
    print("Weighted-Average F1\t", f1BDTwtd)
    BDTAcc.append(accuracyBDT)
    BDTmac.append(f1BDTmac)
    BDTwtd.append(f1BDTwtd)
    outputFile.writelines("Accuracy\t\t" + str(accuracyBDT) + '\n')
    outputFile.writelines("Macro-Average F1\t" + str(f1BDTmac) + '\n')
    outputFile.writelines("Weighted-Average F1\t" + str(f1BDTwtd) + '\n')

    # c) Using Top-DT classifier to predict the split dataset
    #   and will print out the predicted outcome of the Top-DT
    #  classifier
    print("======================================================================")
    print("(a) Top-DT: a better performing Decision Tree")
    outputFile.write("======================================================================\n")
    outputFile.write("(a) Top-DT: a better performing Decision Tree\n")
    param_TopDT = {'criterion': ['gini', 'entropy'],
                   'max_depth': [5, 20],
                   'min_samples_split': [3, 6, 9]}

    classifierTop_DT = GridSearchCV(DecisionTreeClassifier(), param_TopDT, scoring='accuracy')
    classifierTop_DT.fit(x_train, y_train)
    print("Best hyperparameters as selected by GridSearchCV: ")
    print(classifierTop_DT.best_params_)
    outputFile.writelines("Best hyperparameters as selected by GridSearchCV: \n")
    outputFile.writelines(classifierTop_DT.best_params_)

    predictionTop_DT = classifierTop_DT.predict(x_test)
    print("Top-DT Results: ")
    print(predictionTop_DT)
    print("Confusion matrix for top DT: ")
    print("(b) Confusion matrix for top DT: ")
    outputFile.writelines("Top-DT Results: \n")
    outputFile.writelines(str(predictionTop_DT) + '\n')
    outputFile.writelines("(b) Confusion matrix for top DT: \n")
    matrixTDT = confusion_matrix(y_test_array, predictionTop_DT, labels=["drugA", "drugB", "drugC", "drugX", "drugY"])
    print(matrixTDT)
    print("Precision, Recall, and F1-Score per drug")
    print("(c) Precision, Recall, and F1-Score per drug")
    outputFile.writelines(str(matrixTDT) + '\n')
    outputFile.writelines("(c) Precision, Recall, and F1-Score per drug \n")
    precisionTDT, recallTDT, f1TDT, supportTDT = precision_recall_fscore_support(y_test_array, predictionTop_DT,
                                                                                 average=None,
                                                                                 labels=["drugA", "drugB", "drugC",
                                                                                         "drugX",
                                                                                         "drugY"])
    print("\t\t\t\tdrugA\t\tdrugB\t\tdrugC\t\tdrugX\t\tdrugY")
    print("\t\tdrugA\t\tdrugB\t\tdrugC\t\tdrugX\t\tdrugY")
    print("Precision\t", precisionTDT)
    print("Recall\t\t", recallTDT)
    print("F1-Score\t", f1TDT)
    print("Occurences\t", supportTDT)
    print()
    outputFile.writelines("\t\tdrugA\t\tdrugB\t\tdrugC\t\tdrugX\t\tdrugY\n")
    outputFile.writelines("Precision\t" + str(precisionTDT) + '\n')
    outputFile.writelines("Recall\t\t" + str(recallTDT) + '\n')
    outputFile.writelines("F1-Score\t" + str(f1TDT) + '\n')
    outputFile.writelines("Occurences\t" + str(supportTDT) + '\n')
    outputFile.writelines('\n')
    precisionTDTmac, recallTDTmac, f1TDTmac, supportTDTmac = precision_recall_fscore_support(y_test_array,
                                                                                             predictionTop_DT,
                                                                                             average='macro')
    precisionTDTwtd, recallTDTwtd, f1TDTwtd, supportTDTwtd = precision_recall_fscore_support(y_test_array,
                                                                                             predictionTop_DT,
                                                                                             average='weighted')
    accuracyTDT = accuracy_score(y_test_array, predictionTop_DT)
    print("(d) The accuracy, macro-average F1 and weighted-average F1 of the model")
    outputFile.write("(d) The accuracy, macro-average F1 and weighted-average F1 of the model\n")
    print("Accuracy\t\t", accuracyTDT)
    print("Macro-Average F1\t", f1TDTmac)
    print("Weighted-Average F1\t", f1TDTwtd)
    TDTAcc.append(accuracyTDT)
    TDTmac.append(f1TDTmac)
    TDTwtd.append(f1TDTwtd)
    outputFile.writelines("Accuracy\t\t" + str(accuracyTDT) + '\n')
    outputFile.writelines("Macro-Average F1\t" + str(f1TDTmac) + '\n')
    outputFile.writelines("Weighted-Average F1\t" + str(f1TDTwtd) + '\n')

    # d) Using Perceptron classifier to predict the split dataset
    #   and will print out the predicted outcome of the perceptron
    #   classifier
    print("======================================================================")
    outputFile.write("======================================================================\n")
    print("(a) PER: the Perceptron")
    outputFile.write("(a) PER: the Perceptron\n")
    classifierPercep = Perceptron()
    classifierPercep.fit(x_train, y_train)

    predictionPercep = classifierPercep.predict(x_test)
    print("Perceptron Prediction Results: ")
    print(predictionPercep)
    print("Confusion matrix for perceptron: ")
    print("(b) Confusion matrix for perceptron: ")
    outputFile.writelines("Perceptron Prediction Results: \n")
    outputFile.writelines(str(predictionPercep) + '\n')
    outputFile.writelines("(b) Confusion matrix for perceptron: \n")
    matrixPER = confusion_matrix(y_test_array, predictionPercep, labels=["drugA", "drugB", "drugC", "drugX", "drugY"])
    print(matrixPER)
    print("Precision, Recall, and F1-Score per drug")
    print("(c) Precision, Recall, and F1-Score per drug")
    outputFile.writelines(str(matrixPER) + '\n')
    outputFile.writelines("(c) Precision, Recall, and F1-Score per drug\n")
    precisionPER, recallPER, f1PER, supportPER = precision_recall_fscore_support(y_test_array, predictionPercep,
                                                                                 average=None,
                                                                                 labels=["drugA", "drugB", "drugC",
                                                                                         "drugX",
                                                                                         "drugY"])
    print("\t\t\t\tdrugA\t\tdrugB\t\tdrugC\t\tdrugX\t\tdrugY")
    print("\t\tdrugA\t\tdrugB\t\tdrugC\t\tdrugX\t\tdrugY")
    print("Precision\t", precisionPER)
    print("Recall\t\t", recallPER)
    print("F1-Score\t", f1PER)
    print("Occurences\t", supportPER)
    print()
    outputFile.writelines("\t\tdrugA\t\tdrugB\t\tdrugC\t\tdrugX\t\tdrugY\n")
    outputFile.writelines("Precision\t" + str(precisionPER) + '\n')
    outputFile.writelines("Recall\t\t" + str(recallPER) + '\n')
    outputFile.writelines("F1-Score\t" + str(f1PER) + '\n')
    outputFile.writelines("Occurences\t" + str(supportPER) + '\n')
    outputFile.writelines('\n')
    precisionPERmac, recallPERmac, f1PERmac, supportPERmac = precision_recall_fscore_support(y_test_array,
                                                                                             predictionPercep,
                                                                                             average='macro')
    precisionPERwtd, recallPERwtd, f1PERwtd, supportPERwtd = precision_recall_fscore_support(y_test_array,
                                                                                             predictionPercep,
                                                                                             average='weighted')
    accuracyPER = accuracy_score(y_test_array, predictionPercep)
    print("(d) The accuracy, macro-average F1 and weighted-average F1 of the model")
    outputFile.write("(d) The accuracy, macro-average F1 and weighted-average F1 of the model\n")
    print("Accuracy\t\t", accuracyPER)
    print("Macro-Average F1\t", f1PERmac)
    print("Weighted-Average F1\t", f1PERwtd)
    PerAcc.append(accuracyPER)
    Permac.append(f1PERmac)
    Perwtd.append(f1PERwtd)
    outputFile.writelines("Accuracy\t\t" + str(accuracyPER) + '\n')
    outputFile.writelines("Macro-Average F1\t" + str(f1PERmac) + '\n')
    outputFile.writelines("Weighted-Average F1\t" + str(f1PERwtd) + '\n')

    # e) Using Base-MLP classifier to predict the split dataset
    #   and will print out the predicted outcome of the Base-MLP
    #   classifier
    print("======================================================================")
    print("(a) Base-MLP: the Multi-Layered Perceptron")
    outputFile.write("======================================================================\n")
    outputFile.write("(a) Base-MLP: the Multi-Layered Perceptron\n")
    classifierBase_MLP = MLPClassifier(activation='logistic', solver='sgd')
    classifierBase_MLP.fit(x_train, y_train)

    predictionBase_MLP = classifierBase_MLP.predict(x_test)
    print("MLP Prediction Results: ")
    print(predictionBase_MLP)
    print("Confusion matrix for base MLP: ")
    print("(b) Confusion matrix for base MLP: ")
    outputFile.writelines("MLP Prediction Results: \n")
    outputFile.writelines(str(predictionBase_MLP) + '\n')
    outputFile.writelines("(b) Confusion matrix for base MLP: \n")
    matrixBMLP = confusion_matrix(y_test, predictionBase_MLP, labels=["drugA", "drugB", "drugC", "drugX", "drugY"])
    print(matrixBMLP)
    print("Precision, Recall, and F1-Score per drug")
    print("(c) Precision, Recall, and F1-Score per drug")
    outputFile.writelines(str(matrixBMLP) + '\n')
    outputFile.writelines("(c) Precision, Recall, and F1-Score per drug\n")
    precisionBMLP, recallBMLP, f1BMLP, supportBMLP = precision_recall_fscore_support(y_test_array, predictionBase_MLP,
                                                                                     average=None,
                                                                                     labels=["drugA", "drugB", "drugC",
                                                                                             "drugX",
                                                                                             "drugY"])
    print("\t\t\t\tdrugA\t\tdrugB\t\tdrugC\t\tdrugX\t\tdrugY")
    print("\t\tdrugA\t\tdrugB\t\tdrugC\t\tdrugX\t\tdrugY")
    print("Precision\t", precisionBMLP)
    print("Recall\t\t", recallBMLP)
    print("F1-Score\t", f1BMLP)
    print("Occurences\t", supportBMLP)
    print()
    outputFile.writelines("\t\tdrugA\t\tdrugB\t\tdrugC\t\tdrugX\t\tdrugY\n")
    outputFile.writelines("Precision\t" + str(precisionBMLP) + '\n')
    outputFile.writelines("Recall\t\t" + str(recallBMLP) + '\n')
    outputFile.writelines("F1-Score\t" + str(f1BMLP) + '\n')
    outputFile.writelines("Occurences\t" + str(supportBMLP) + '\n')
    outputFile.writelines('\n')
    precisionBMLPmac, recallBMLPmac, f1BMLPmac, supportBMLPmac = precision_recall_fscore_support(y_test_array,
                                                                                                 predictionBase_MLP,
                                                                                                 average='macro')
    precisionBMLPwtd, recallBMLPwtd, f1BMLPwtd, supportBMLPwtd = precision_recall_fscore_support(y_test_array,
                                                                                                 predictionBase_MLP,
                                                                                                 average='weighted')
    accuracyBMLP = accuracy_score(y_test_array, predictionBase_MLP)
    print("(d) The accuracy, macro-average F1 and weighted-average F1 of the model")
    outputFile.write("(d) The accuracy, macro-average F1 and weighted-average F1 of the model\n")
    print("Accuracy\t\t", accuracyBMLP)
    print("Macro-Average F1\t", f1BMLPmac)
    print("Weighted-Average F1\t", f1BMLPwtd)
    BMLPAcc.append(accuracyBMLP)
    BMLPmac.append(f1BMLPmac)
    BMLPwtd.append(f1BMLPwtd)
    outputFile.writelines("Accuracy\t\t" + str(accuracyBMLP) + '\n')
    outputFile.writelines("Macro-Average F1\t" + str(f1BMLPmac) + '\n')
    outputFile.writelines("Weighted-Average F1\t" + str(f1BMLPwtd) + '\n')

    # f) Using Top-MLP classifier to predict the split dataset
    #   and will print out the predicted outcome of the Top-MLP
    #   classifier
    print("======================================================================")
    print("(a) Top-MLP: a better performing Multi-Layered Perceptron")
    outputFile.write("======================================================================\n")
    outputFile.write("(a) Top-MLP: a better performing Multi-Layered Perceptron\n")
    param_TopMLP = {'activation': ['tanh', 'relu', 'identity'],
                    'hidden_layer_sizes': [(30, 50, 150), (8, 24, 116)],
                    'solver': ['adam', 'sgd']
                    }

    classifierTop_MLP = GridSearchCV(estimator=MLPClassifier(max_iter=200), param_grid=param_TopMLP, n_jobs=-1,
                                     scoring='accuracy')
    classifierTop_MLP.fit(x_train, y_train)
    print("Best hyperparameters as selected by GridSearchCV: ")
    print(classifierTop_DT.best_params_)
    outputFile.writelines("Best hyperparameters as selected by GridSearchCV: \n")
    outputFile.writelines(str(classifierTop_DT.best_params_) + '\n')

    predictionTop_MLP = classifierTop_MLP.predict(x_test)
    print("Top-MLP Prediction Results: ")
    print(predictionTop_MLP)
    print("Confusion matrix for top MLP: ")
    print("(b) Confusion matrix for top MLP: ")
    outputFile.writelines("Top-MLP Prediction Results: \n")
    outputFile.writelines(str(predictionTop_MLP) + '\n')
    outputFile.writelines("(b) Confusion matrix for top MLP: \n")
    matrixTMLP = confusion_matrix(y_test_array, predictionTop_MLP, labels=["drugA", "drugB", "drugC", "drugX", "drugY"])
    print(matrixTMLP)
    print("Precision, Recall, and F1-Score per drug")
    print("(c) Precision, Recall, and F1-Score per drug")
    outputFile.writelines(str(matrixTMLP) + '\n')
    outputFile.writelines("(c) Precision, Recall, and F1-Score per drug\n")
    precisionTMLP, recallTMLP, f1TMLP, supportTMLP = precision_recall_fscore_support(y_test_array, predictionTop_MLP,
                                                                                     average=None,
                                                                                     labels=["drugA", "drugB", "drugC",
                                                                                             "drugX",
                                                                                             "drugY"])
    print("\t\t\t\tdrugA\t\tdrugB\t\tdrugC\t\tdrugX\t\tdrugY")
    print("\t\tdrugA\t\tdrugB\t\tdrugC\t\tdrugX\t\tdrugY")
    print("Precision\t", precisionTMLP)
    print("Recall\t\t", recallTMLP)
    print("F1-Score\t", f1TMLP)
    print("Occurences\t", supportTMLP)
    print()
    outputFile.writelines("\t\tdrugA\t\tdrugB\t\tdrugC\t\tdrugX\t\tdrugY\n")
    outputFile.writelines("Precision\t" + str(precisionTMLP) + '\n')
    outputFile.writelines("Recall\t\t" + str(recallTMLP) + '\n')
    outputFile.writelines("F1-Score\t" + str(f1TMLP) + '\n')
    outputFile.writelines("Occurences\t" + str(supportTMLP) + '\n')
    outputFile.writelines('\n')
    precisionTMLPmac, recallTMLPmac, f1TMLPmac, supportTMLPmac = precision_recall_fscore_support(y_test_array,
                                                                                                 predictionTop_MLP,
                                                                                                 average='macro')
    precisionTMLPwtd, recallTMLPwtd, f1TMLPwtd, supportTMLPwtd = precision_recall_fscore_support(y_test_array,
                                                                                                 predictionTop_MLP,
                                                                                                 average='weighted')
    accuracyTMLP = accuracy_score(y_test_array, predictionTop_MLP)
    print("(d) The accuracy, macro-average F1 and weighted-average F1 of the model")
    outputFile.write("(d) The accuracy, macro-average F1 and weighted-average F1 of the model\n")
    print("Accuracy\t\t", accuracyTMLP)
    print("Macro-Average F1\t", f1TMLPmac)
    print("Weighted-Average F1\t", f1TMLPwtd)
    print("Weighted-Average F1\t", f1TMLPwtd)
    TMLPAcc.append(accuracyTMLP)
    TMLPmac.append(f1TMLPmac)
    TMLPwtd.append(f1TMLPwtd)
    outputFile.writelines("Accuracy\t\t" + str(accuracyTMLP) + '\n')
    outputFile.writelines("Macro-Average F1\t" + str(f1TMLPmac) + '\n')
    outputFile.writelines("Weighted-Average F1\t" + str(f1TMLPwtd) + '\n')

    # Task 2 Part 8, run the code 10 times and get the average metrics requested.

    NBAcc, NBmac, NBwtd = list(), list(), list()
    BDTAcc, BDTmac, BDTwtd = list(), list(), list()
    TDTAcc, TDTmac, TDTwtd = list(), list(), list()
    PerAcc, Permac, Perwtd = list(), list(), list()
    BMLPAcc, BMLPmac, BMLPwtd = list(), list(), list()
    TMLPAcc, TMLPmac, TMLPwtd = list(), list(), list()

    for i in range(10):
        # a) Using Gaussian Naive Bayes classifier to predict the split dataset
        #   and will print out the predicted outcome of the Naive Bayes
        #   classifier
        classifierNB = GaussianNB()
        classifierNB.fit(x_train, y_train)

        predictionNB = classifierNB.predict(x_test)
        precisionNBmac, recallNBmac, f1NBmac, supportNBmac = precision_recall_fscore_support(y_test_array, predictionNB,
                                                                                             average='macro')
        precisionNBwtd, recallNBwtd, f1NBwtd, supportNBwtd = precision_recall_fscore_support(y_test_array, predictionNB,
                                                                                             average='weighted')
        accuracyNB = accuracy_score(y_test_array, predictionNB)
        NBAcc.append(accuracyNB)
        NBmac.append(f1NBmac)
        NBwtd.append(f1NBwtd)

        # b) using Decision Tree classifier to predict the split dataset
        #   and will print out the predicted outcome of the Base-DT
        #   classifier
        classifierBase_DT = DecisionTreeClassifier()
        classifierBase_DT.fit(x_train, y_train)

        predictionBase_DT = classifierBase_DT.predict(x_test)
        precisionBDTmac, recallBDTmac, f1BDTmac, supportBDTmac = precision_recall_fscore_support(y_test_array,
                                                                                                 predictionBase_DT,
                                                                                                 average='macro')
        precisionBDTwtd, recallBDTwtd, f1BDTwtd, supportBDTwtd = precision_recall_fscore_support(y_test_array,
                                                                                                 predictionBase_DT,
                                                                                                 average='weighted')
        accuracyBDT = accuracy_score(y_test_array, predictionBase_DT)
        BDTAcc.append(accuracyBDT)
        BDTmac.append(f1BDTmac)
        BDTwtd.append(f1BDTwtd)

        # c) Using Top-DT classifier to predict the split dataset
        #   and will print out the predicted outcome of the Top-DT
        #  classifier
        param_TopDT = {'criterion': ['gini', 'entropy'],
                       'max_depth': [5, 20],
                       'min_samples_split': [3, 6, 9]}

        classifierTop_DT = GridSearchCV(DecisionTreeClassifier(), param_TopDT, scoring='accuracy')
        classifierTop_DT.fit(x_train, y_train)
        predictionTop_DT = classifierTop_DT.predict(x_test)
        precisionTDTmac, recallTDTmac, f1TDTmac, supportTDTmac = precision_recall_fscore_support(y_test_array,
                                                                                                 predictionTop_DT,
                                                                                                 average='macro')
        precisionTDTwtd, recallTDTwtd, f1TDTwtd, supportTDTwtd = precision_recall_fscore_support(y_test_array,
                                                                                                 predictionTop_DT,
                                                                                                 average='weighted')
        accuracyTDT = accuracy_score(y_test_array, predictionTop_DT)
        TDTAcc.append(accuracyTDT)
        TDTmac.append(f1TDTmac)
        TDTwtd.append(f1TDTwtd)

        # d) Using Perceptron classifier to predict the split dataset
        #   and will print out the predicted outcome of the perceptron
        #   classifier
        classifierPercep = Perceptron()
        classifierPercep.fit(x_train, y_train)

        predictionPercep = classifierPercep.predict(x_test)
        precisionPERmac, recallPERmac, f1PERmac, supportPERmac = precision_recall_fscore_support(y_test_array,
                                                                                                 predictionPercep,
                                                                                                 average='macro')
        precisionPERwtd, recallPERwtd, f1PERwtd, supportPERwtd = precision_recall_fscore_support(y_test_array,
                                                                                                 predictionPercep,
                                                                                                 average='weighted')
        accuracyPER = accuracy_score(y_test_array, predictionPercep)
        PerAcc.append(accuracyPER)
        Permac.append(f1PERmac)
        Perwtd.append(f1PERwtd)

        # e) Using Base-MLP classifier to predict the split dataset
        #   and will print out the predicted outcome of the Base-MLP
        #   classifier
        classifierBase_MLP = MLPClassifier(activation='logistic', solver='sgd')
        classifierBase_MLP.fit(x_train, y_train)

        predictionBase_MLP = classifierBase_MLP.predict(x_test)
        precisionBMLPmac, recallBMLPmac, f1BMLPmac, supportBMLPmac = precision_recall_fscore_support(y_test_array,
                                                                                                     predictionBase_MLP,
                                                                                                     average='macro')
        precisionBMLPwtd, recallBMLPwtd, f1BMLPwtd, supportBMLPwtd = precision_recall_fscore_support(y_test_array,
                                                                                                     predictionBase_MLP,
                                                                                                     average='weighted')
        accuracyBMLP = accuracy_score(y_test_array, predictionBase_MLP)
        BMLPAcc.append(accuracyBMLP)
        BMLPmac.append(f1BMLPmac)
        BMLPwtd.append(f1BMLPwtd)

        # f) Using Top-MLP classifier to predict the split dataset
        #   and will print out the predicted outcome of the Top-MLP
        #   classifier
        param_TopMLP = {'activation': ['tanh', 'relu', 'identity'],
                        'hidden_layer_sizes': [(30, 50, 150), (8, 24, 116)],
                        'solver': ['adam', 'sgd']
                        }

        classifierTop_MLP = GridSearchCV(estimator=MLPClassifier(max_iter=200), param_grid=param_TopMLP, n_jobs=-1,
                                         scoring='accuracy')
        classifierTop_MLP.fit(x_train, y_train)
        predictionTop_MLP = classifierTop_MLP.predict(x_test)
        precisionTMLPmac, recallTMLPmac, f1TMLPmac, supportTMLPmac = precision_recall_fscore_support(y_test_array,
                                                                                                     predictionTop_MLP,
                                                                                                     average='macro')
        precisionTMLPwtd, recallTMLPwtd, f1TMLPwtd, supportTMLPwtd = precision_recall_fscore_support(y_test_array,
                                                                                                     predictionTop_MLP,
                                                                                                     average='weighted')
        accuracyTMLP = accuracy_score(y_test_array, predictionTop_MLP)
        TMLPAcc.append(accuracyTMLP)
        TMLPmac.append(f1TMLPmac)
        TMLPwtd.append(f1TMLPwtd)

    print("======================================================================")
    outputFile.write("======================================================================\n")
    print("\n\n\nTask 2 Part 8\n")
    outputFile.write("\n\n\nTask 2 Part 8\n")

    NBAccAvg = mean(NBAcc)
    NBwtdAvg = mean(NBwtd)
    NBmacAvg = mean(NBmac)
    NBAccStD = statistics.stdev(NBAcc)
    NBwtdStD = statistics.stdev(NBwtd)
    NBmacStD = statistics.stdev(NBmac)

    print("(a) NB:")
    print("\tAverage Accuracy: ", NBAccAvg)
    print("\tStandard Deviation for Accuracy: ", NBAccStD)
    print("\tAverage Macro-Average F1: ", NBmacAvg)
    print("\tStandard Deviation for Macro-Average F1: ", NBmacStD)
    print("\tAverage Weighted-Average F1: ", NBwtdAvg)
    print("\tStandard Deviation for Weighted-Average F1: ", NBwtdStD)
    print()
    outputFile.writelines("(a) NB:\n")
    outputFile.writelines("\tAverage Accuracy: " + str(NBAccAvg) + '\n')
    outputFile.writelines("\tStandard Deviation for Accuracy: " + str(NBAccStD) + '\n')
    outputFile.writelines("\tAverage Macro-Average F1: " + str(NBmacAvg) + '\n')
    outputFile.writelines("\tStandard Deviation for Macro-Average F1: " + str(NBmacStD) + '\n')
    outputFile.writelines("\tAverage Weighted-Average F1: " + str(NBwtdAvg) + '\n')
    outputFile.writelines("\tStandard Deviation for Weighted-Average F1: " + str(NBwtdStD) + '\n')
    outputFile.writelines('\n')

    BDTAccAvg = mean(BDTAcc)
    BDTwtdAvg = mean(BDTwtd)
    BDTmacAvg = mean(BDTmac)
    BDTAccStD = statistics.stdev(BDTAcc)
    BDTwtdStD = statistics.stdev(BDTwtd)
    BDTmacStD = statistics.stdev(BDTmac)

    print("(b) Base-DT:")
    print("\tAverage Accuracy: ", BDTAccAvg)
    print("\tStandard Deviation for Accuracy: ", BDTAccStD)
    print("\tAverage Macro-Average F1: ", BDTmacAvg)
    print("\tStandard Deviation for Macro-Average F1: ", BDTmacStD)
    print("\tAverage Weighted-Average F1: ", BDTwtdAvg)
    print("\tStandard Deviation for Weighted-Average F1: ", BDTwtdStD)
    print()
    outputFile.writelines("(b) Base-DT:\n")
    outputFile.writelines("\tAverage Accuracy: " + str(BDTAccAvg) + '\n')
    outputFile.writelines("\tStandard Deviation for Accuracy: " + str(BDTAccStD) + '\n')
    outputFile.writelines("\tAverage Macro-Average F1: " + str(BDTmacAvg) + '\n')
    outputFile.writelines("\tStandard Deviation for Macro-Average F1: " + str(BDTmacStD) + '\n')
    outputFile.writelines("\tAverage Weighted-Average F1: " + str(BDTwtdAvg) + '\n')
    outputFile.writelines("\tStandard Deviation for Weighted-Average F1: " + str(BDTwtdStD) + '\n')
    outputFile.writelines('\n')

    TDTAccAvg = mean(TDTAcc)
    TDTwtdAvg = mean(TDTwtd)
    TDTmacAvg = mean(TDTmac)
    TDTAccStD = statistics.stdev(TDTAcc)
    TDTwtdStD = statistics.stdev(TDTwtd)
    TDTmacStD = statistics.stdev(TDTmac)

    print("(c) Top-DT:")
    print("\tAverage Accuracy: ", TDTAccAvg)
    print("\tStandard Deviation for Accuracy: ", TDTAccStD)
    print("\tAverage Macro-Average F1: ", TDTmacAvg)
    print("\tStandard Deviation for Macro-Average F1: ", TDTmacStD)
    print("\tAverage Weighted-Average F1: ", TDTwtdAvg)
    print("\tStandard Deviation for Weighted-Average F1: ", TDTwtdStD)
    print()
    outputFile.writelines("(c) Top-DT:\n")
    outputFile.writelines("\tAverage Accuracy: " + str(TDTAccAvg) + '\n')
    outputFile.writelines("\tStandard Deviation for Accuracy: " + str(TDTAccStD) + '\n')
    outputFile.writelines("\tAverage Macro-Average F1: " + str(TDTmacAvg) + '\n')
    outputFile.writelines("\tStandard Deviation for Macro-Average F1: " + str(TDTmacStD) + '\n')
    outputFile.writelines("\tAverage Weighted-Average F1: " + str(TDTwtdAvg) + '\n')
    outputFile.writelines("\tStandard Deviation for Weighted-Average F1: " + str(TDTwtdStD) + '\n')
    outputFile.writelines('\n')

    PerAccAvg = mean(PerAcc)
    PerwtdAvg = mean(Perwtd)
    PermacAvg = mean(Permac)
    PerAccStD = statistics.stdev(PerAcc)
    PerwtdStD = statistics.stdev(Perwtd)
    PermacStD = statistics.stdev(Permac)

    print("(d) PER:")
    print("\tAverage Accuracy: ", PerAccAvg)
    print("\tStandard Deviation for Accuracy: ", PerAccStD)
    print("\tAverage Macro-Average F1: ", PermacAvg)
    print("\tStandard Deviation for Macro-Average F1: ", PermacStD)
    print("\tAverage Weighted-Average F1: ", PerwtdAvg)
    print("\tStandard Deviation for Weighted-Average F1: ", PerwtdStD)
    print()
    outputFile.writelines("(d) PER:\n")
    outputFile.writelines("\tAverage Accuracy: " + str(PerAccAvg) + '\n')
    outputFile.writelines("\tStandard Deviation for Accuracy: " + str(PerAccStD) + '\n')
    outputFile.writelines("\tAverage Macro-Average F1: " + str(PermacAvg) + '\n')
    outputFile.writelines("\tStandard Deviation for Macro-Average F1: " + str(PermacStD) + '\n')
    outputFile.writelines("\tAverage Weighted-Average F1: " + str(PerwtdAvg) + '\n')
    outputFile.writelines("\tStandard Deviation for Weighted-Average F1: " + str(PerwtdStD) + '\n')
    outputFile.writelines('\n')

    BMLPAccAvg = mean(BMLPAcc)
    BMLPwtdAvg = mean(BMLPwtd)
    BMLPmacAvg = mean(BMLPmac)
    BMLPAccStD = statistics.stdev(BMLPAcc)
    BMLPwtdStD = statistics.stdev(BMLPwtd)
    BMLPmacStD = statistics.stdev(BMLPmac)

    print("(e) Base-MLP:")
    print("\tAverage Accuracy: ", BMLPAccAvg)
    print("\tStandard Deviation for Accuracy: ", BMLPAccStD)
    print("\tAverage Macro-Average F1: ", BMLPmacAvg)
    print("\tStandard Deviation for Macro-Average F1: ", BMLPmacStD)
    print("\tAverage Weighted-Average F1: ", BMLPwtdAvg)
    print("\tStandard Deviation for Weighted-Average F1: ", BMLPwtdStD)
    print()
    outputFile.writelines("(e) Base-MLP:\n")
    outputFile.writelines("\tAverage Accuracy: " + str(BMLPAccAvg) + '\n')
    outputFile.writelines("\tStandard Deviation for Accuracy: " + str(BMLPAccStD) + '\n')
    outputFile.writelines("\tAverage Macro-Average F1: " + str(BMLPmacAvg) + '\n')
    outputFile.writelines("\tStandard Deviation for Macro-Average F1: " + str(BMLPmacStD) + '\n')
    outputFile.writelines("\tAverage Weighted-Average F1: " + str(BMLPwtdAvg) + '\n')
    outputFile.writelines("\tStandard Deviation for Weighted-Average F1: " + str(BMLPwtdStD) + '\n')
    outputFile.writelines('\n')

    TMLPAccAvg = mean(TMLPAcc)
    TMLPwtdAvg = mean(TMLPwtd)
    TMLPmacAvg = mean(TMLPmac)
    TMLPAccStD = statistics.stdev(TMLPAcc)
    TMLPwtdStD = statistics.stdev(TMLPwtd)
    TMLPmacStD = statistics.stdev(TMLPmac)

    print("(f) Top-MLP:")
    print("\tAverage Accuracy: ", TMLPAccAvg)
    print("\tStandard Deviation for Accuracy: ", TMLPAccStD)
    print("\tAverage Macro-Average F1: ", TMLPmacAvg)
    print("\tStandard Deviation for Macro-Average F1: ", TMLPmacStD)
    print("\tAverage Weighted-Average F1: ", TMLPwtdAvg)
    print("\tStandard Deviation for Weighted-Average F1: ", TMLPwtdStD)
    print()
    outputFile.writelines("(f) Top-MLP:\n")
    outputFile.writelines("\tAverage Accuracy: " + str(TMLPAccAvg) + '\n')
    outputFile.writelines("\tStandard Deviation for Accuracy: " + str(TMLPAccStD) + '\n')
    outputFile.writelines("\tAverage Macro-Average F1: " + str(TMLPmacAvg) + '\n')
    outputFile.writelines("\tStandard Deviation for Macro-Average F1: " + str(TMLPmacStD) + '\n')
    outputFile.writelines("\tAverage Weighted-Average F1: " + str(TMLPwtdAvg) + '\n')
    outputFile.writelines("\tStandard Deviation for Weighted-Average F1: " + str(TMLPwtdStD) + '\n')
    outputFile.writelines('\n')

    outputFile.close()