from sklearn import model_selection, tree
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from sys import argv
from os import path


if not path.is_file(arv[1]):
    print("Error: could not find specified CSV dataset")

training_data = pd.read_csv(argv[1])
training_data = pd.get_dummies(training_data)
training_data_arr = training_data.to_numpy()

classifier = tree.DecisionTreeClassifier(max_depth=4)
classifier = classifier.fit(training_data_arr[:, 1:], training_data_arr[:, 0])

    gr_class = GradientBoostingClassifier()
    gr_class = gr_class.fit(X, Y)
    return gr_class


def kfold(training_data_arr, n_folds):

    kf = model_selection.KFold(n_splits=n_folds)
    i = 0
    j = 1
    max_score_grad = 0
    gr_score = 0
    gr_class = GradientBoostingClassifier()
    for train, test in kf.split(training_data_arr):
        gr_class = gr_class.fit(training_data_arr[train, 1:], training_data_arr[train, 0])
        gr_score = gr_class.score(training_data_arr[test, 1:], training_data_arr[test, 0])
        print("Gradient Boost Score " + str(j) + ": " + str(gr_score))

        if gr_score > max_score_grad:
            best_class_gr = gr_class
        j += 1
        i += gr_score

    mean_score = i / n_folds

    return best_class_gr, mean_score


def main():
    if not path.isfile(argv[1]):
        print("Error: could not find specified CSV dataset for training")

    training_data = pd.read_csv(argv[1])
    training_data = training_data.sample(frac=1)
    training_data = pd.get_dummies(training_data)
    training_data_arr = training_data.to_numpy()
    single_classifier = train_tree(training_data_arr)
    folded_classifier, score = kfold(training_data_arr, 3)

    print("10-fold Classifier mean score: " + str(score))

"""
        dot_data = tree.export_graphviz(t, out_file=(str(index) + "graph.dot"),
                                        feature_names=training_data.columns.to_list()[1:],
                                        class_names=["True", "False"],
                                        filled=True,
                                        rounded=True)
"""

main()
