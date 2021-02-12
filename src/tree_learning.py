from sklearn import model_selection, tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from sys import argv
from os import path


def train_tree(training_data_arr):

    X = training_data_arr[:, 1:]
    Y = training_data_arr[:, 0]

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

        print("--------MODEL " + str(j) + " QUALITY--------")
        true_y = training_data_arr[test, 0]
        pred_y = gr_class.predict(training_data_arr[test, 1:])
        print_scores(true_y=true_y, pred_y=pred_y, beta=2.0)

        if gr_score > max_score_grad:
            best_class_gr = gr_class
        j += 1
        i += gr_score

    mean_score = i / n_folds

    return best_class_gr, mean_score


def print_scores(true_y, pred_y, beta=1.0):
    """

    Parameters
    ----------
    model Modello di cui calcolare le principali metriche
    true_y Vettore dei valori veri per la feature di classe
    pred_y Vettore dei valori predetti per la feature di classe

    Returns
    -------

    """

    (pr, rec, f_sc, su) = precision_recall_fscore_support(y_true=true_y, y_pred=pred_y, beta=beta, average="macro")
    acc = accuracy_score(y_true=true_y, y_pred=pred_y)
    print("Accuracy:\t" + str(acc))
    print("Precision:\t" + str(pr))
    print("Recall:\t\t" + str(rec))
    print("F-measure" + ":\t" + str(f_sc))
    print("(beta " + str(beta) + ")")


def main():
    if not path.isfile(argv[1]):
        print("Error: could not find specified CSV dataset for training")

    try:
        n_folds = int(argv[2])
        if n_folds <= 0:
            print("Error: specified a non positive fold value")
    except IndexError:
        n_folds = 10

    training_data = pd.read_csv(argv[1])
    training_data = training_data.sample(frac=1)
    training_data = pd.get_dummies(training_data)
    training_data_arr = training_data.to_numpy()
    single_classifier = train_tree(training_data_arr)
    folded_classifier, score = kfold(training_data_arr, n_folds)
    print("\n----MEAN SCORE----")
    print(str(n_folds) + "-folds classifier mean score: " + str(score))

"""
        dot_data = tree.export_graphviz(t, out_file=(str(index) + "graph.dot"),
                                        feature_names=training_data.columns.to_list()[1:],
                                        class_names=["True", "False"],
                                        filled=True,
                                        rounded=True)
"""

main()
