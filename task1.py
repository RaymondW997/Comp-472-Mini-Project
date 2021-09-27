import numpy as np
import sklearn.datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def print_hi():
    bbc = sklearn.datasets.load_files(container_path='dataset/BBC', encoding='latin1')

    # Q3: Get all category names
    print("Q3: Get all category names")
    for x in bbc.target_names:
        print(x)

    print("===========================")
    print("Divide dataset into 80training/20testing")
    bbc_train_set, bbc_test_set = train_test_split(bbc.data, test_size=0.25, random_state=None)

    print("===========================")
    # Q4 Preprocess the data, tokenize the corpus
    print("Q4 Preprocess the data, tokenize the corpus")
    count_vect = CountVectorizer()
    bbc_train_counts = count_vect.fit_transform(bbc_train_set)

    print(bbc_train_counts.shape)
    print("===========================")
    print("count word 'dollar' occurancy:")
    print(count_vect.vocabulary_.get(u'dollar'))

    print("===========================")
    print("Term frequencies")
    tf_transformer = TfidfTransformer(use_idf=False).fit(bbc_train_counts)
    bbc_train_tf = tf_transformer.transform(bbc_train_counts)
    print("Term frequencies -> tf")
    print(bbc_train_tf.shape)

    tfidf_transformer = TfidfTransformer()
    bbc_train_tfidf = tfidf_transformer.fit_transform(bbc_train_counts)
    print("Term frequencies -> tf-idf")
    print(bbc_train_tfidf.shape)

    print("===========================")
    print("Training a classifier")
    clf = MultinomialNB().fit(bbc_train_tfidf, bbc_train_set)

    print("===========================")
    print("Evaluation of the performance on the test set")
    bbc_test_counts = count_vect.transform(bbc_test_set)
    bbc_test_tfidf = tfidf_transformer.transform(bbc_test_counts)
    predicted = clf.predict(bbc_test_tfidf)
    # np.mean(predicted == bbc_test_set)
    # print(predicted)
    docs_new = ["dollar", "asian"]
    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    for doc, category in zip(docs_new, predicted):
        print('%r => %s' % (doc, bbc.target_names[category]))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi()
