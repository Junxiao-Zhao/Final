import thulac, csv, random, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score

def classify(test, p0, p1, p2):
    l = [sum(test*p0), sum(test*p1), sum(test*p2)]
    closest = max(l)
    return l.index(closest)

def acc(vocab, p0, p1, p2):
    print("ACCing...")
    thulac.thu1 = thulac.thulac(seg_only=True)
    pd_all = pd.read_csv('simplifyweibo_4_moods.csv')
    test_data = list()
    test_category = list()
    for moodes in range(3):
        for lines in random.sample(list(pd_all[pd_all.label == moodes]["review"])[5000:], 100):
            test_category.append(moodes)
            test_data.append(thulac.thu1.cut(lines, text=True).split(" "))
    pred_category = list()
    for each in test_data:
        l = np.array(match(vocab, each))
        pred_category.append(classify(l, p0, p1, p2))
    print(pred_category)
    print(accuracy_score(test_category, pred_category))

def train(matrix, category):
    print("Training...")
    num_words = len(matrix[0])
    p0_num = np.ones(num_words)
    p1_num = np.ones(num_words)
    p2_num = np.ones(num_words)
    p0_denom = 2
    p1_denom = 2
    p2_denom = 2
    for i in range(len(matrix)):
        if (category[i] == 0):
            p0_num += matrix[i]
            p0_denom += sum(matrix[i])
        if (category[i] == 1):
            p1_num += matrix[i]
            p1_denom += sum(matrix[i])
        else:
            p2_num += matrix[i]
            p2_denom += sum(matrix[i])
    p0 = p0_num / p0_denom
    p1 = p1_num / p1_denom
    p2 = p2_num / p2_denom
    return p0, p1, p2


def match(vocab, text):
    l = [0] * len(vocab)
    for words in text:
        if words in vocab:
            l[vocab.index(words)] = 1
        else:
            print("The word: %s is not in my Vocabulary!" % words)
    return l

def nodup(data):
    print("Eliminating duplicate words...")
    vocab = set([])
    for document in data:
        vocab = vocab | set(document)
    return list(vocab)

def read_csv(filename):
    print("Reading files...")
    thulac.thu1 = thulac.thulac(seg_only=True)
    pd_all = pd.read_csv('simplifyweibo_4_moods.csv')
    data = list()
    category = list()
    for moodes in range(3):
        for lines in list(pd_all[pd_all.label == moodes]["review"])[0:5000]:
            category.append(moodes)
            data.append(thulac.thu1.cut(lines, text=True).split(" "))
    return data, category

def main():
    data, category = read_csv("simplifyweibo_4_moods.csv")
    vocab = nodup(data)
    matrix = []
    for texts in data:
        matrix.append(match(vocab, texts))
    p0, p1, p2 = train(np.array(matrix), np.array(category))
    #print(p0, p1, p2)
    acc(vocab, p0, p1, p2)
    
main()