from sklearn import model_selection, tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error, mean_squared_error, accuracy_score,\
    max_error
import pandas as pd
from sys import argv
from os import path
import pickle


def k_fold(training_data_arr, n_folds, x_start, x_end, y_index, is_regressor=False):

    kf = model_selection.KFold(n_splits=n_folds)
    i = 0
    j = 1
    max_score = 0
    curr_score = 0
    if is_regressor:
        curr_tree = DecisionTreeRegressor(max_depth=5)
    else:
        curr_tree = GradientBoostingClassifier()

    for train, test in kf.split(training_data_arr):
        print("Fold " + str(j) + "/" + str(n_folds))
        curr_tree = curr_tree.fit(training_data_arr[train, x_start:x_end], training_data_arr[train, y_index])
        curr_score = curr_tree.score(training_data_arr[test, x_start:x_end], training_data_arr[test, y_index])

        print("--------MODEL " + str(j) + " QUALITY--------")
        true_y = training_data_arr[test, y_index]
        pred_y = curr_tree.predict(training_data_arr[test, x_start:x_end])

        if not is_regressor:
            print_classifier_scores(true_y=true_y, pred_y=pred_y, beta=2.0)
        else:
            print_regressor_scores(true_y=true_y, pred_y=pred_y)

        if curr_score > max_score:
            best_tree = curr_tree
            max_score = curr_score

        j += 1
        i += curr_score

    mean_score = i / n_folds

    return best_tree, mean_score


def print_regressor_scores(true_y, pred_y):
    mse = mean_squared_error(true_y, pred_y)
    mae = mean_absolute_error(true_y, pred_y)
    max = max_error(true_y, pred_y)
    print("Mean Squared Error:\t\t" + str(mse))
    print("Mean Absolute Error:\t" + str(mae))
    print("Max Error:\t\t\t\t" + str(max))


def print_classifier_scores(true_y, pred_y, beta=1.0):
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
        return data_frame["OriginalLeadTime"] - data_frame["LeadTime"]


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
    notice_days = cancellation_minus_arrival(data_frame=training_data)
    training_data = training_data.drop(["OriginalLeadTime"], axis=1)

    training_data = training_data.sample(frac=1)
    training_data = pd.get_dummies(training_data)
    training_data_arr = training_data.to_numpy()
    best_classifier, score = k_fold(training_data_arr, n_folds, 1, training_data_arr.shape[1], 0)
    print("\n----MEAN SCORE----")
    print(str(n_folds) + "-folds classifier mean score: " + str(score))

    # Apprendimento di un albero per la previsione del numero di giorni che trascorreranno prima della cancellazione
    # di una prenotazione
    to_drop = training_data[(training_data["IsCanceled"] == 0)].index
    training_data_canceled = training_data.copy()
    training_data_canceled["CancellationMinusArrival"] = notice_days
    training_data_canceled = training_data_canceled.drop(to_drop, axis=0)
    training_data_canceled = training_data_canceled.drop("IsCanceled", axis=1)
    training_data_canceled_arr = training_data_canceled.to_numpy()

    best_regressor, reg_score = k_fold(training_data_canceled_arr,
                                                         n_folds,
                                                         0, -1,
                                                         training_data_canceled_arr.shape[1] - 1,
                                                         is_regressor=True)

    print("\n----MEAN SCORE----")
    print(str(n_folds) + "-folds classifier mean score: " + str(reg_score))

    target_names_classification = training_data["IsCanceled"].unique().tolist()

    first_dot = tree.export_graphviz(best_classifier.estimators_[42, 0],
                feature_names=training_data.columns.tolist()[1:],
                class_names=target_names_classification,
                filled=True,
                rounded=True,
                out_file="..\\res\\" + path.basename(argv[1]).replace(".csv", "") + "_canceled_classifier_42.dot")

    target_names_regression = training_data_canceled["CancellationMinusArrival"].unique().tolist()

    second_dot = tree.export_graphviz(best_regressor,
                feature_names=training_data_canceled.columns.tolist()[:-1],
                class_names=target_names_regression,
                filled=True,
                rounded=True,
                out_file="..\\res\\" + path.basename(argv[1]).replace(".csv", "") + "_notice_days_regressor.dot")

    # Serializzazione dei modelli
    classifier_path = "..\\models\\" + path.basename(argv[1].replace(".csv", "_classifier.tree"))
    with open(classifier_path, "wb") as out_classifier:
        pickle.dump(best_classifier, out_classifier)

    regressor_path = "..\\models\\" + path.basename(argv[1].replace(".csv", "_regressor.tree"))
    with open(regressor_path, "wb") as out_regressor:
        pickle.dump(best_regressor,  out_regressor)


main()
