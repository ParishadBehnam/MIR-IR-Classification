import pickle

from Main import search_kNN, search_Naive, save_docs, train_Naive, train_svm, search_svm, train_RF, search_RF
from Main import find_best_k, find_best_c, P_R_per_category
from Main import load_docs

while True:
    command = input()
    if command == "quit":
        break

    elif command == "save":
        l = save_docs()
        with open('train_docs.pkl', 'wb') as outp:
            pickle.dump(l, outp)
        print("Done")

    elif command == "search naive":
        with open("hamshahri/Test/"+input("query?")+".txt") as q:
            all_text = q.read()
            query = all_text[(all_text.find("text :") + 7):]
            with open('train_naive.pkl', 'rb') as inp:
                file = pickle.load(inp)
                print(search_Naive(query, file))

    elif command == "search knn":
        k = input("which k?")
        with open("hamshahri/Test/"+input("query?")+".txt") as q:
            all_text = q.read()
            query = all_text[(all_text.find("text :") + 7):]
            print(search_kNN(k, query, load_docs()))

    elif command == "train naive":
        train_Naive()
        print("Done.")

    elif command == "train svm":
        with open('train_docs.pkl', 'rb') as inp:
            l = pickle.load(inp)
        t = train_svm(float(input("c?")), l)
        with open('train_svm.pkl', 'wb') as outp:
            pickle.dump(t, outp)
        print("Done.")

    elif command == "search svm":
        with open('train_svm.pkl', 'rb') as inp:
            l = pickle.load(inp)
        with open("hamshahri/Test/"+input("query?")+".txt") as q:
            all_text = q.read()
            query = all_text[(all_text.find("text :") + 7):]
            print(str(search_svm(query, l)))

    elif command == "train RF":
        with open('train_docs.pkl', 'rb') as inp:
            l = pickle.load(inp)
        t = train_RF(l)
        with open('train_RF.pkl', 'wb') as outp:
            pickle.dump(t, outp)
        print("Done.")

    elif command == "search RF":
        with open('train_RF.pkl', 'rb') as inp:
            l = pickle.load(inp)
        with open("hamshahri/Test/"+input("query?")+".txt") as q:
            all_text = q.read()
            query = all_text[(all_text.find("text :") + 7):]
            print(str(search_RF(query, l)))

    elif command == "find best k":
        k, p = find_best_k([1, 5, 10])
        print(str(k) + " is the best k with precision of " + str(p))

    elif command == "find best c":
        c, p = find_best_c([0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
        print(str(c) + " is the best c with precision of " + str(p))

    elif command == "evaluate":
        alg = input("which algorithm? naive / knn / svm / RF?")
        print(P_R_per_category(0 if alg == "naive" else 1 if alg == "knn" else 2 if alg == "svm" else 3))

    else:
        print("invalid command!")
