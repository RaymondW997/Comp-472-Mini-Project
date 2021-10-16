import sklearn.datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import matplotlib.pyplot as plt
import os


def print_hi():
    bbc = sklearn.datasets.load_files(container_path='dataset/BBC', encoding='latin1')
    output_log = open('bbc-performance.txt', 'w')
    # Q3: Get all category names
    print("Q3: Get all category names")
    print(bbc.target_names)

    # Iterate all folder and count the files under the categories
    instances = []
    for target_name in bbc.target_names:
        path, dirs, files = next(os.walk("dataset/BBC/" + target_name))
        file_count = len(files)
        instances.append(file_count)

    plt.title("Distribution of the instances in each class")
    plt.xlabel("Class name")
    plt.ylabel("Number of instances")
    plt.bar(np.array(bbc.target_names), np.array(instances))
    plt.savefig('BBC-Distribution.pdf')
    plt.show()

    print("===========================")
    print("Divide dataset into 80% training / 20% testing")
    x_train, x_test, y_train, y_test = train_test_split(bbc.data, bbc.target, test_size=0.2, train_size=0.8,
                                                        random_state=None)

    # MultinomialNB
    print("===========================", file=output_log)
    print("===========MultinomialNB default values, try 1============", file=output_log)
    print("===========================")
    # Q4 Preprocess the data, tokenize the corpus
    print("Q4 Building pipeline. Preprocess the data, tokenize the corpus")
    from sklearn.pipeline import Pipeline
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB(alpha=1)),
    ])

    print("===========================", file=output_log)
    print("Training a classifier", file=output_log)
    text_clf.fit(x_train, y_train)

    print("===========================", file=output_log)
    print("Evaluation of the performance on the test set", file=output_log)

    predicted = text_clf.predict(x_test)
    y_probability = np.mean(predicted == y_test)
    print(y_probability, file=output_log)

    print("===========================", file=output_log)
    print("(b) the confusion matrix", file=output_log)
    from sklearn import metrics
    confusion = metrics.confusion_matrix(y_test, predicted)
    print(confusion, file=output_log)

    print("===========================", file=output_log)
    print("(c) the precision, recall, and F1-measure for each class", file=output_log)
    print(metrics.classification_report(y_test, predicted), file=output_log)

    print("===========================", file=output_log)
    print("(d) the accuracy, macro-average F1 and weighted-average F1 of the model", file=output_log)
    acc = metrics.accuracy_score(y_test, predicted)
    print("Accuracy is :" + str(acc), file=output_log)
    macro_avg = metrics.f1_score(y_test, predicted, average="macro")
    print("Macro avg is :" + str(macro_avg), file=output_log)
    weighted_avg = metrics.f1_score(y_test, predicted, average="weighted")
    print("Weighted avg is :" + str(weighted_avg), file=output_log)

    print("===========================", file=output_log)
    print("(e) the prior probability of each class", file=output_log)
    nb = MultinomialNB(alpha=1)
    separate_cat_vec = CountVectorizer()
    separate_cat_vec.fit(x_train)
    x_train_transformed = separate_cat_vec.transform(x_train)
    nb.fit(x_train_transformed, y_train)
    print(nb.classes_, file=output_log)
    print(nb.class_log_prior_, file=output_log)

    print("===========================", file=output_log)
    print("(f) the size of the vocabulary (i.e. the number of different words)", file=output_log)
    cat_vec = CountVectorizer()
    cat_array = cat_vec.fit_transform(bbc.data).toarray()
    cat_sum = cat_array.sum()
    size_voc = len(cat_vec.get_feature_names())
    print(size_voc, file=output_log)

    # g: the number of word-tokens in each class
    print("===========================", file=output_log)
    print("(g) the number of word-tokens in each class (i.e. the number of words in total)", file=output_log)
    for target_name in bbc.target_names:
        separate_cat = sklearn.datasets.load_files(container_path='dataset/BBC', categories=target_name,
                                                   encoding='latin1')
        separate_cat_data = separate_cat.data
        separate_cat_vec = CountVectorizer()
        separate_cat_array = separate_cat_vec.fit_transform(separate_cat_data).toarray()
        separate_cat_sum = separate_cat_array.sum()
        print("The number of word-tokens in " + target_name + ":" + str(separate_cat_sum), file=output_log)

    print("===========================", file=output_log)
    print("(h) the number of word-tokens in the entire corpus", file=output_log)
    print(cat_sum, file=output_log)

    print("============================", file=output_log)
    print("(i) the number and percentage of words with a frequency of zero in each class", file=output_log)
    for target_name in bbc.target_names:
        separate_cat = sklearn.datasets.load_files(container_path='dataset/BBC', categories=target_name,
                                                   encoding='latin1')
        separate_cat_data = separate_cat.data
        separate_cat_vec = CountVectorizer()
        separate_cat_vec.fit(separate_cat_data)
        current_voc_count = len(separate_cat_vec.get_feature_names())
        count_zero = size_voc - current_voc_count
        print("the number words with a frequency of zero in class " + target_name + ": " + str(count_zero),
              file=output_log)
        print("the percentage of words with a frequency of zero in class " + target_name + ": " + str(
            count_zero / size_voc), file=output_log)

    print("===========================", file=output_log)
    print("(j) the number and percentage of words with a frequency of one in the entire corpus", file=output_log)
    count = 0
    for value in cat_array.sum(axis=0):
        if value == 1:
            count = count + 1
    print("The number of words with a frequency of 1 in entire corpus: " + str(count), file=output_log)
    print("The percentage: " + str(count / cat_sum), file=output_log)

    # MultinomialNB 3
    print("===========================", file=output_log)
    print("===========MultinomialNB default values, try 2============", file=output_log)
    print("===========================")
    # Q4 Preprocess the data, tokenize the corpus
    print("Q4 Building pipeline. Preprocess the data, tokenize the corpus")
    from sklearn.pipeline import Pipeline
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB(alpha=0.9)),
    ])

    print("===========================", file=output_log)
    print("Training a classifier", file=output_log)
    text_clf.fit(x_train, y_train)

    print("===========================", file=output_log)
    print("Evaluation of the performance on the test set", file=output_log)

    predicted = text_clf.predict(x_test)
    y_probability = np.mean(predicted == y_test)
    print(y_probability, file=output_log)

    print("===========================", file=output_log)
    print("(b) the confusion matrix", file=output_log)
    from sklearn import metrics
    confusion = metrics.confusion_matrix(y_test, predicted)
    print(confusion, file=output_log)

    print("===========================", file=output_log)
    print("(c) the precision, recall, and F1-measure for each class", file=output_log)
    print(metrics.classification_report(y_test, predicted), file=output_log)

    print("===========================", file=output_log)
    print("(d) the accuracy, macro-average F1 and weighted-average F1 of the model", file=output_log)
    acc = metrics.accuracy_score(y_test, predicted)
    print("Accuracy is :" + str(acc), file=output_log)
    macro_avg = metrics.f1_score(y_test, predicted, average="macro")
    print("Macro avg is :" + str(macro_avg), file=output_log)
    weighted_avg = metrics.f1_score(y_test, predicted, average="weighted")
    print("Weighted avg is :" + str(weighted_avg), file=output_log)

    print("===========================", file=output_log)
    print("(e) the prior probability of each class", file=output_log)
    nb = MultinomialNB(alpha=1)
    separate_cat_vec = CountVectorizer()
    separate_cat_vec.fit(x_train)
    x_train_transformed = separate_cat_vec.transform(x_train)
    nb.fit(x_train_transformed, y_train)
    print(nb.classes_, file=output_log)
    print(nb.class_log_prior_, file=output_log)

    print("===========================", file=output_log)
    print("(f) the size of the vocabulary (i.e. the number of different words)", file=output_log)
    cat_vec = CountVectorizer()
    cat_array = cat_vec.fit_transform(bbc.data).toarray()
    cat_sum = cat_array.sum()
    size_voc = len(cat_vec.get_feature_names())
    print(size_voc, file=output_log)

    # g: the number of word-tokens in each class
    print("===========================", file=output_log)
    print("(g) the number of word-tokens in each class (i.e. the number of words in total)", file=output_log)
    for target_name in bbc.target_names:
        separate_cat = sklearn.datasets.load_files(container_path='dataset/BBC', categories=target_name,
                                                   encoding='latin1')
        separate_cat_data = separate_cat.data
        separate_cat_vec = CountVectorizer()
        separate_cat_array = separate_cat_vec.fit_transform(separate_cat_data).toarray()
        separate_cat_sum = separate_cat_array.sum()
        print("The number of word-tokens in " + target_name + ":" + str(separate_cat_sum), file=output_log)

    print("===========================", file=output_log)
    print("(h) the number of word-tokens in the entire corpus", file=output_log)
    print(cat_sum, file=output_log)

    print("============================", file=output_log)
    print("(i) the number and percentage of words with a frequency of zero in each class", file=output_log)
    for target_name in bbc.target_names:
        separate_cat = sklearn.datasets.load_files(container_path='dataset/BBC', categories=target_name,
                                                   encoding='latin1')
        separate_cat_data = separate_cat.data
        separate_cat_vec = CountVectorizer()
        separate_cat_vec.fit(separate_cat_data)
        current_voc_count = len(separate_cat_vec.get_feature_names())
        count_zero = size_voc - current_voc_count
        print("the number words with a frequency of zero in class " + target_name + ": " + str(count_zero),
              file=output_log)
        print("the percentage of words with a frequency of zero in class " + target_name + ": " + str(
            count_zero / size_voc), file=output_log)

    print("===========================", file=output_log)
    print("(j) the number and percentage of words with a frequency of one in the entire corpus", file=output_log)
    count = 0
    for value in cat_array.sum(axis=0):
        if value == 1:
            count = count + 1
    print("The number of words with a frequency of 1 in entire corpus: " + str(count), file=output_log)
    print("The percentage: " + str(count / cat_sum), file=output_log)

    # MultinomialNB 2
    print("===========================", file=output_log)
    print("===========MultinomialNB default values, try 2============", file=output_log)
    print("===========================")
    # Q4 Preprocess the data, tokenize the corpus
    print("Q4 Building pipeline. Preprocess the data, tokenize the corpus")
    from sklearn.pipeline import Pipeline
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB(alpha=0.0001)),
    ])

    print("===========================", file=output_log)
    print("Training a classifier", file=output_log)
    text_clf.fit(x_train, y_train)

    print("===========================", file=output_log)
    print("Evaluation of the performance on the test set", file=output_log)

    predicted = text_clf.predict(x_test)
    y_probability = np.mean(predicted == y_test)
    print(y_probability, file=output_log)

    print("===========================", file=output_log)
    print("(b) the confusion matrix", file=output_log)
    from sklearn import metrics
    confusion = metrics.confusion_matrix(y_test, predicted)
    print(confusion, file=output_log)

    print("===========================", file=output_log)
    print("(c) the precision, recall, and F1-measure for each class", file=output_log)
    print(metrics.classification_report(y_test, predicted), file=output_log)

    print("===========================", file=output_log)
    print("(d) the accuracy, macro-average F1 and weighted-average F1 of the model", file=output_log)
    acc = metrics.accuracy_score(y_test, predicted)
    print("Accuracy is :" + str(acc), file=output_log)
    macro_avg = metrics.f1_score(y_test, predicted, average="macro")
    print("Macro avg is :" + str(macro_avg), file=output_log)
    weighted_avg = metrics.f1_score(y_test, predicted, average="weighted")
    print("Weighted avg is :" + str(weighted_avg), file=output_log)

    print("===========================", file=output_log)
    print("(e) the prior probability of each class", file=output_log)
    nb = MultinomialNB(alpha=1)
    separate_cat_vec = CountVectorizer()
    separate_cat_vec.fit(x_train)
    x_train_transformed = separate_cat_vec.transform(x_train)
    nb.fit(x_train_transformed, y_train)
    print(nb.classes_, file=output_log)
    print(nb.class_log_prior_, file=output_log)

    print("===========================", file=output_log)
    print("(f) the size of the vocabulary (i.e. the number of different words)", file=output_log)
    cat_vec = CountVectorizer()
    cat_array = cat_vec.fit_transform(bbc.data).toarray()
    cat_sum = cat_array.sum()
    size_voc = len(cat_vec.get_feature_names())
    print(size_voc, file=output_log)

    # g: the number of word-tokens in each class
    print("===========================", file=output_log)
    print("(g) the number of word-tokens in each class (i.e. the number of words in total)", file=output_log)
    for target_name in bbc.target_names:
        separate_cat = sklearn.datasets.load_files(container_path='dataset/BBC', categories=target_name,
                                                   encoding='latin1')
        separate_cat_data = separate_cat.data
        separate_cat_vec = CountVectorizer()
        separate_cat_array = separate_cat_vec.fit_transform(separate_cat_data).toarray()
        separate_cat_sum = separate_cat_array.sum()
        print("The number of word-tokens in " + target_name + ":" + str(separate_cat_sum), file=output_log)

    print("===========================", file=output_log)
    print("(h) the number of word-tokens in the entire corpus", file=output_log)
    print(cat_sum, file=output_log)

    print("============================", file=output_log)
    print("(i) the number and percentage of words with a frequency of zero in each class", file=output_log)
    for target_name in bbc.target_names:
        separate_cat = sklearn.datasets.load_files(container_path='dataset/BBC', categories=target_name,
                                                   encoding='latin1')
        separate_cat_data = separate_cat.data
        separate_cat_vec = CountVectorizer()
        separate_cat_vec.fit(separate_cat_data)
        current_voc_count = len(separate_cat_vec.get_feature_names())
        count_zero = size_voc - current_voc_count
        print("the number words with a frequency of zero in class " + target_name + ": " + str(count_zero),
              file=output_log)
        print("the percentage of words with a frequency of zero in class " + target_name + ": " + str(
            count_zero / size_voc), file=output_log)

    print("===========================", file=output_log)
    print("(j) the number and percentage of words with a frequency of one in the entire corpus", file=output_log)
    count = 0
    for value in cat_array.sum(axis=0):
        if value == 1:
            count = count + 1
    print("The number of words with a frequency of 1 in entire corpus: " + str(count), file=output_log)
    print("The percentage: " + str(count / cat_sum), file=output_log)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi()
