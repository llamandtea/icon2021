from sklearn import model_selection, tree
from pydotplus import graph_from_dot_data
from IPython.display import Image, display
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from matplotlib import pyplot as plt
import pandas as pd
from sys import argv
from os import path


def train_tree(training_data_arr):

    X = training_data_arr[:, 1:]
    Y = training_data_arr[:, 0]

    gr_class = GradientBoostingClassifier()
    gr_class = gr_class.fit(X, Y)
    return gr_class


def train_regression_tree(training_data_arr, n_col):
    X = training_data_arr[:, :(n_col-2)]
    Y = training_data_arr[:, (n_col-1)]

    grr_class = GradientBoostingRegressor()
    grr_class = grr_class.fit(X, Y)
    return grr_class


def kfold(training_data_arr, n_folds, x_start, x_end, y_index, is_regressor=False):

    kf = model_selection.KFold(n_splits=n_folds)
    i = 0
    j = 1
    max_score_grad = 0
    gr_score = 0
    if is_regressor:
        gr_class = GradientBoostingRegressor()
    else:
        gr_class = GradientBoostingClassifier()

    for train, test in kf.split(training_data_arr):
        gr_class = gr_class.fit(training_data_arr[train, x_start:x_end], training_data_arr[train, y_index])
        gr_score = gr_class.score(training_data_arr[test, x_start:x_end], training_data_arr[test, y_index])

        print("--------MODEL " + str(j) + " QUALITY--------")
        true_y = training_data_arr[test, y_index]
        pred_y = gr_class.predict(training_data_arr[test, x_start:x_end])
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


def cancellation_minus_arrival(data_frame):
        data_frame["CancellationMinusArrival"] = data_frame.apply(lambda x:   x["OriginalLeadTime"]-x["LeadTime"], axis=1)
        return data_frame


def main():
    if not path.isfile(argv[1]):
        print("Error: could not find specified CSV dataset for training")
        return

    try:
        n_folds = int(argv[2])
        if n_folds <= 0:
            print("Error: specified a non positive fold value")
    except IndexError:
        n_folds = 10

    training_data = pd.read_csv(argv[1])

    # Aggiunta della colonna che demarca lo scarto fra il giorno di cancellazione e l'originale giorno di arrivo
    training_data = cancellation_minus_arrival(training_data)
    training_data = training_data.drop(["OriginalLeadTime"], axis=1)

    training_data = training_data.sample(frac=1)
    training_data = pd.get_dummies(training_data)
    training_data_arr = training_data.to_numpy()
    single_classifier = train_tree(training_data_arr)
    folded_classifier, score = kfold(training_data_arr, n_folds, 1, len(training_data.columns.tolist()) - 1, 0)
    print("\n----MEAN SCORE----")
    print(str(n_folds) + "-folds classifier mean score: " + str(score))

    # Apprendimento di un albero per la previsione del numero di giorni che trascorreranno prima della cancellazione
    # di una prenotazione
    to_drop = training_data[(training_data["IsCanceled"] == 0)].index
    training_data_canceled = training_data.drop(to_drop, axis=0)
    training_data_canceled_arr = training_data_canceled.to_numpy()
    canc_minus_arrival_regressor = train_regression_tree(training_data_canceled_arr, len(training_data_canceled.columns.tolist()))
    best_canc_minus_arrival_regressor, reg_score = kfold(training_data_arr, n_folds, 0, len(training_data.columns.tolist()) - 2,
                                              len(training_data.columns.tolist()) - 1)

    print("\n----MEAN SCORE----")
    print(str(n_folds) + "-folds classifier mean score: " + str(reg_score))


    target_names_classification = training_data["IsCanceled"].unique().tolist()
    first_dot = tree.export_graphviz(folded_classifier.estimators_[42, 0],
                feature_names=training_data.columns.tolist()[2:],
                class_names=target_names_classification,
                filled=True,
                rounded=True,
                out_file="..\\res\\" + path.basename(argv[1]).replace(".csv", "") + "_canceled_classifier_42.dot")

    target_names_regression = training_data["CancellationMinusArrival"].unique().tolist()

    second_dot = tree.export_graphviz(best_canc_minus_arrival_regressor.estimators_[42, 0],
                feature_names=training_data.columns.tolist()[1: len(training_data_canceled.columns.tolist())-1],
                class_names=target_names_regression,
                filled=True,
                rounded=True,
                out_file="..\\res\\" + path.basename(argv[1]).replace(".csv", "") + "_days_regressor_42.dot")


main()
