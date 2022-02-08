import thulac, csv, random, numpy as np, pandas as pd
from NBClassifier import *
from sklearn.metrics import accuracy_score

def read_csv(filename):
    print("Reading files...")
    thulac.thu1 = thulac.thulac(seg_only=True)
    pd_all = pd.read_csv(filename)
    data = list()
    category = list()
    for moodes in range(3):
        for lines in list(pd_all[pd_all.label == moodes]["review"])[0:5000]:
            category.append(moodes)
            data.append(thulac.thu1.cut(lines, text=True).split(" "))
    return data, category

def main():
    filename = "weibo_senti_100k.csv"
    data, category = read_csv(filename)
    nb = NBayes()
    nb.train_set(data, category)
    print("ACCing...")
    thulac.thu1 = thulac.thulac(seg_only=True)
    pd_all = pd.read_csv(filename)
    test_data = list()
    test_category = list()
    for moodes in range(len(set(category))):
        for lines in random.sample(list(pd_all[pd_all.label == moodes]["review"])[5000:], 100):
            test_category.append(moodes)
            test_data.append(thulac.thu1.cut(lines, text=True).split(" "))
    pred_category = list()
    for each in test_data:
        nb.map2vocab(each)
        pred_category.append(nb.predict(nb.testset))
    print(accuracy_score(test_category, pred_category))

main()