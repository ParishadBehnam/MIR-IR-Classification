import pickle
import heapq

import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from hazm import *


def save_docs(docs_num=2000):
    stemmer = Stemmer()
    normalizer = Normalizer()
    token_num_per_cat = {}
    category_per_doc = [""] * docs_num

    tokens = {}     # after using tfifVectorizer it's no need to use this dictionary!
    tokens_doc = [None] * docs_num
    corpus = [""] * docs_num

    for docID in range(0, docs_num):
        with open("hamshahri/Train/" + str(docID+1) + ".txt", 'r+', encoding="utf-8") as doc:
            all = (word_tokenize(normalizer.normalize(doc.read())))
            words = all[all.index("text") + 2:]
            cat = all[all.index("category") + 2]
            if cat not in token_num_per_cat:
                token_num_per_cat[cat] = 0
            category_per_doc[docID] = cat
            tokens_doc[docID] = {}
            for w in words:
                token_num_per_cat[cat] += 1
                token = stemmer.stem(w)
                if token not in tokens:
                    tokens[token] = 0
                if token not in tokens_doc[docID]:
                    tokens_doc[docID][token] = 1
                    tokens[token] += 1
                else:
                    tokens_doc[docID][token] += 1
            s = ""
            for w in words:
                s += (stemmer.stem(w) + " ")
            corpus[docID] = s

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    idf = vectorizer.idf_
    d = (dict(zip(vectorizer.get_feature_names(), idf)))    #mapping tokens to idfs.
    return [category_per_doc, token_num_per_cat, tokens_doc, d]


def load_docs():
    with open('train_docs.pkl', 'rb') as inp:
        f = pickle.load(inp)
    return f


def train_Naive():
    l = load_docs()
    category_per_doc = l[0]
    token_num_per_cat = l[1]
    token_per_doc = l[2]
    idfs = l[3]
    doc_num_per_cat = {}
    vector_per_cat = {}

    for docID in range(0, (len(category_per_doc))):
        cat = category_per_doc[docID]
        if cat in doc_num_per_cat:
            doc_num_per_cat[cat] += 1
        else:
            doc_num_per_cat[cat] = 1
            vector_per_cat[cat] = {x: 0 for x in idfs.keys()}

        for t in idfs.keys():
            if t in token_per_doc[docID]:
                vector_per_cat[cat][t] += float(idfs[t]) * token_per_doc[docID][t]  # multiplying tf in idf per category and token.

    out = [doc_num_per_cat, vector_per_cat, len(category_per_doc), token_num_per_cat, idfs]
    with open('train_naive.pkl', 'wb') as outp:
        pickle.dump(out, outp)


def search_Naive(q, file):
    # train_Naive()

    normalizer = Normalizer()
    stemmer = Stemmer()
    doc_num_per_cat = file[0]
    vector_per_cat = file[1]
    docs_num = file[2]
    token_num_per_cat = file[3]
    idfs = file[4]
    cats = doc_num_per_cat.keys()

    words = word_tokenize(normalizer.normalize(q))
    # print(words)

    maximum = float('-inf')
    out = ""
    for cat in cats:
        s = math.log10((float(doc_num_per_cat[cat])) / docs_num)
        for w in words:
            t = stemmer.stem(w)
            if t in idfs:
                s += math.log10(((float(vector_per_cat[cat][t]) + 1) / float(token_num_per_cat[cat] + len(idfs))))  #linear smoothing:D
            else:
                s += math.log10(float(1) / (token_num_per_cat[cat] + len(idfs)))
        if s > maximum:
            maximum = s
            out = cat

    return out


def train_kNN(k):   # no train time! all is done in save_docs() function
    pass


def search_kNN(k, q, l):

    category_per_doc = l[0]
    token_per_doc = l[2]
    idfs = l[3]

    k = int(k)
    normalizer = Normalizer()
    stemmer = Stemmer()

    sort = []

    words = word_tokenize(normalizer.normalize(q))
    vec = {x: 0 for x in idfs.keys()}
    for w in words:
        t = stemmer.stem(w)
        if t in idfs:
            vec[t] += 1

    for doc in range(0, len(category_per_doc)):
        s = 0
        for w in words:
            t = stemmer.stem(w)
            if t in idfs and t in token_per_doc[doc]:
                s += idfs[t] * token_per_doc[doc][t] * vec[t]
        sort.append(((-1)*s, doc))      # in order to pop easily the minimum element of heap.

    heapq.heapify(sort)
    ks = {}
    for _ in range(0, k):
        l = (heapq.heappop(sort)[1])
        if category_per_doc[l] not in ks:
            ks[category_per_doc[l]] = 1
        else:
            ks[category_per_doc[l]] += 1
    m = 0
    mcat = 0
    for i,j in ks.items():
        if j > m:
            m = j
            mcat = i
    return mcat


def train_svm(c, l):
    # l = load_docs()
    category_per_doc = l[0]
    token_per_doc = l[2]
    idfs = l[3]
    token_id = {}
    docs_num = len(category_per_doc)
    X = [None] * docs_num
    i = 0
    for t in idfs.keys():
        token_id[t] = i
        i += 1
    for d in range(0, docs_num):
        X[d] = [None] * len(idfs)
        for t in idfs.keys():
            if t in token_per_doc[d]:
                X[d][token_id[t]] = token_per_doc[d][t] * idfs[t]
            else:
                X[d][token_id[t]] = 0
    y = [category_per_doc[d] for d in range(0, docs_num)]
    clf = LinearSVC()
    clf.fit(X, y)
    LinearSVC(C=c, class_weight=None, max_iter=-1, random_state=None, tol=0.001, verbose=False)

    return [clf, idfs, token_id]


def search_svm(q, l):
    clf = l[0]
    idfs = l[1]
    token_id = l[2]

    normalizer = Normalizer()
    stemmer = Stemmer()
    ws = word_tokenize(normalizer.normalize(q))
    words = {}
    for i in range(0, len(ws)):
        t = stemmer.stem(ws[i])
        if t in words:
            words[t] += 1
        else:
            words[t] = 1

    X = [None] * len(idfs)
    for t in idfs.keys():
        if t in words:
            X[token_id[t]] = words[t] * idfs[t]
        else:
            X[token_id[t]] = 0
    return clf.predict([X])[0]


def find_best_k(ks):
    num = int(0.9*2000)
    normalizer = Normalizer()
    stemmer = Stemmer()
    l = save_docs(docs_num=num)
    maximum = 0
    maxk = 0
    corrects = 0
    for k in ks:
        corrects = 0
        for docID in range(num, 2000):  # validation docs
            with open("hamshahri/Train/" + str(docID+1) + ".txt", 'r+', encoding="utf-8") as doc:
                all = (word_tokenize(normalizer.normalize(doc.read())))
                words = all[all.index("text") + 2:]
                cat = all[all.index("category") + 2]
                s = ""
                for w in words:
                    s += (stemmer.stem(w) + " ")
                if cat == search_kNN(k, s, l):  # true positive!
                    corrects += 1
        if corrects > maximum:
            maximum = corrects
            maxk = k
        # print(k, corrects)
    return maxk, corrects/float(0.1*2000)


def find_best_c(cs):
    num = int(0.9*2000)
    normalizer = Normalizer()
    stemmer = Stemmer()
    docs = save_docs(docs_num=num)
    maximum = 0
    maxk = 0
    corrects = 0
    for c in cs:
        corrects = 0
        l = train_svm(c, docs)
        for docID in range(num, 2000):
            with open("hamshahri/Train/" + str(docID + 1) + ".txt", 'r+', encoding="utf-8") as doc:
                all = (word_tokenize(normalizer.normalize(doc.read())))
                words = all[all.index("text") + 2:]
                cat = all[all.index("category") + 2]
                s = ""
                for w in words:
                    s += (stemmer.stem(w) + " ")
                if cat == search_svm(s, l):
                    corrects += 1
        if corrects > maximum:
            maximum = corrects
            maxk = c
        # print(c, corrects)
    return maxk, corrects/float(0.1*2000)


def P_R_per_category(alg):
    normalizer = Normalizer()
    stemmer = Stemmer()
    actual_cat = {}
    predicted_cat = {}
    correct_per_cat = {}

    if alg == 0:
        with open('train_naive.pkl', 'rb') as inp:
            l = pickle.load(inp)
    elif alg == 1:
        l = load_docs()
    elif alg == 2:
        with open('train_svm.pkl', 'rb') as inp:
            l = pickle.load(inp)
    else:
        with open('train_RF.pkl', 'rb') as inp:
            l = pickle.load(inp)

    for docID in range(0, 2500):
        if docID < 2000:
            folder = "Train"
        else:
            folder = "Test"
        with open("hamshahri/" + folder + "/" + str(docID + 1) + ".txt", 'r+', encoding="utf-8") as doc:
            all = (word_tokenize(normalizer.normalize(doc.read())))
            words = all[all.index("text") + 2:]
            cat = all[all.index("category") + 2]
            # print(docID)
            s = ""
            for w in words:
                s += (stemmer.stem(w) + " ")
            if alg == 0:
                my_cat = search_Naive(s, l)
            elif alg == 1:
                my_cat = search_kNN(10, s, l)
            elif alg == 2:
                my_cat = search_svm(s, l)
            else:
                my_cat = search_RF(s, l)

            if cat not in actual_cat:
                actual_cat[cat] = 1
            else:
                actual_cat[cat] += 1

            if my_cat not in predicted_cat:
                predicted_cat[my_cat] = 1
            else:
                predicted_cat[my_cat] += 1

            if my_cat == cat:
                if cat not in correct_per_cat:
                    correct_per_cat[cat] = 1
                else:
                    correct_per_cat[cat] += 1

    cor = sum(correct_per_cat.values())
    all_predicts = sum(predicted_cat.values())

    out_p = {}
    out_r = {}
    f1 = {}
    for c, val in correct_per_cat.items():
        out_p[c] = float(correct_per_cat[c]) / predicted_cat[c]     # precision
        out_r[c] = float(correct_per_cat[c]) / actual_cat[c]        # recall
    for c in correct_per_cat.keys():
        f1[c] = (out_p[c] * out_r[c] * 2) / float(out_p[c] + out_r[c])  # F1 computing by recall and precision
    F1 = sum(f1.values()) / len(f1)     # macro averaging

    return {"Precision": out_p, "Recall": out_r, "Accuracy": (float(cor) / all_predicts), "F1": F1}


def train_RF(l):
    category_per_doc = l[0]
    token_per_doc = l[2]
    idfs = l[3]

    token_id = {}
    docs_num = len(category_per_doc)
    X = [None] * docs_num
    i = 0
    for t in idfs.keys():
        token_id[t] = i
        i += 1
    for d in range(0, docs_num):
        X[d] = [None] * len(idfs)
        for t in idfs.keys():
            if t in token_per_doc[d]:
                X[d][token_id[t]] = token_per_doc[d][t] * idfs[t]
            else:
                X[d][token_id[t]] = 0

    y = [category_per_doc[d] for d in range(0, docs_num)]
    clf = RandomForestClassifier(max_depth=20, random_state=0)
    clf.fit(X, y)
    return [clf, idfs, token_id]


def search_RF(q, l):
    clf = l[0]
    idfs = l[1]
    token_id = l[2]

    normalizer = Normalizer()
    stemmer = Stemmer()
    ws = word_tokenize(normalizer.normalize(q))
    words = {}
    # tfvectorizing the query
    for i in range(0, len(ws)):
        t = stemmer.stem(ws[i])
        if t in words:
            words[t] += 1
        else:
            words[t] = 1

    X = [None] * len(idfs)
    # making feature vectors!
    for t in idfs.keys():
        # print(token_id[t])
        if t in words:
            X[token_id[t]] = words[t] * idfs[t]
        else:
            X[token_id[t]] = 0
    return clf.predict([X])[0]





