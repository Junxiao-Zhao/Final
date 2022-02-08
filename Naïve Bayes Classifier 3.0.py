import thulac, csv, random, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

def acc(train_data, train_category, test_data, test_category):
    nb_model = MultinomialNB(alpha=0.001)
    nb_model.fit(train_data, train_category)
    nb_predict = nb_model.predict(test_data)
    print("The accuracy of Multinomial Naïve Bayes text classification is: ", accuracy_score(nb_predict, test_category))

    ber_model = BernoulliNB(alpha=0.001)
    ber_model.fit(train_data, train_category)
    ber_predict = ber_model.predict(test_data)
    print("The accuracy of Bernoulli Naïve Bayes text classification is: ", accuracy_score(ber_predict, test_category))


def tfidf(data, test_data):
    print("'tf-idf'ing...")
    document = [" ".join(each) for each in data]
    test_document = [" ".join(each) for each in test_data]
    tfidf_model = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    train = tfidf_model.fit_transform(document)
    test = tfidf_model.transform(test_document)
    return train, test

def read_csv(filename):
    print("Reading files...")
    thulac.thu1 = thulac.thulac(seg_only=True)
    pd_all = pd.read_csv(filename)
    data = list()
    category = list()
    test_data = list()
    test_category = list()
    for moodes in range(3):
        cur_list = list(pd_all[pd_all.label == moodes]["review"])
        for lines in cur_list[0:int(len(cur_list)/10)]:
            category.append(moodes)
            data.append(thulac.thu1.cut(lines, text=True).split(" "))
        for lines in random.sample(cur_list[int(len(cur_list)/10):], int(len(cur_list)/500)):
            test_category.append(moodes)
            test_data.append(thulac.thu1.cut(lines, text=True).split(" "))
    return data, category, test_data, test_category

def main():
    data, category, test_data, test_category = read_csv("simplifyweibo_4_moods.csv")
    train, test = tfidf(data, test_data)
    acc(train, category, test, test_category)

main()