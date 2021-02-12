"""
preprocessing dei dati, con discretizzazione degli attributi continui
ed encoding delle features categoriche
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import cluster
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
from os import path
from sys import argv


def refactor_data_frame(data_frame):
    """
    Funzione che, preso in input il DataFrame contenente i dati da clusterizzare
    converte tutte le feature continue prima in split e poi in feature indicatrici
    restituisce inoltre il dataframe modificato e il nome delle colonne prima dell'applicazione del dummy
    """

    data_frame = data_frame.assign(Staying=data_frame["StaysInWeekendNights"] + data_frame["StaysInWeekNights"])

    """
    Discretizzazione delle colonne continue in intervalli n-ari. Per ogni colonna sono stati scelti degli estremi
    significativi quando possibile.
    """
    x_adr = pd.cut(data_frame["ADR"], 3)
    x_lead = pd.cut(data_frame["LeadTime"], [0, 7, 30, 90, 180, 365, data_frame["LeadTime"].max()], include_lowest=True)
    x_stays = pd.cut(data_frame["Staying"], [0, 3, 7, 15, data_frame["Staying"].max()], include_lowest=True)
    x_booking_ch = pd.cut(data_frame["BookingChanges"], [0, 5, 10, data_frame["BookingChanges"].max()], include_lowest=True)
    x_adults = pd.cut(data_frame["Adults"], [0,1,data_frame["Adults"].max()], include_lowest=True)
    x_cancel_rate = pd.cut(data_frame["CancelRate"], 2)
    x_minors = pd.cut(data_frame["Minors"], [0, 2, data_frame["Minors"].max()], include_lowest=True)

    # Eliminazione delle colonne convertite dal dataframe originale
    data_frame = data_frame.drop(["ADR",
                                  "LeadTime",
                                  "StaysInWeekendNights",
                                  "StaysInWeekNights",
                                  "BookingChanges",
                                  "Adults",
                                  "CancelRate",
                                  "Minors",
                                 "MarketSegment"], axis=1)

    # Sostituzione delle colonne eliminate con quelle discretizzate
    data_frame = data_frame.assign(ADR=x_adr)
    data_frame = data_frame.assign(LeadTime=x_lead)
    data_frame = data_frame.assign(BookingChanges=x_booking_ch)
    data_frame = data_frame.assign(Adults=x_adults)
    data_frame = data_frame.assign(CancelRate=x_cancel_rate)
    data_frame = data_frame.assign(Minors=x_minors)
    data_frame = data_frame.assign(Staying=x_stays)

    # Conversione del dataframe in dummy encoding
    features = data_frame.columns.tolist()
    data_frame = pd.get_dummies(data_frame)

    return data_frame, features


def k_means_random_restart(data_frame, iter, nclusters):
    """
    Funzione che effettua il clustering attraverso quanto messo a disposizione dalla libreria sklearn per ottenere
    nclusters cluster. Iter indica, invece, il numero random restart da rieseguire per ottenere l'ottimo locale.
    """
    print("Starting clustering...")

    # Inizializzazione del miglior cluster
    best_cluster = cluster.KMeans(n_clusters=nclusters, random_state=np.random.randint(1, 40000)).fit(data_frame)

    for i in range(0, iter):
        current_cluster = cluster.KMeans(n_clusters=nclusters, random_state=np.random.randint(1, 40000)).fit(data_frame)
        if best_cluster.inertia_ > current_cluster.inertia_ :
            best_cluster = current_cluster
        if (i + 1) % 10 == 0:
            print("Iteration number " + str(i + 1))

    print("Best inertia: " + str(best_cluster.inertia_))
    return best_cluster


def dummy_inversion(data_frame, features, kmeans, centroid_flag=True):
    """
    preso in input il dataframe, le features e e quanto ottenuto dal clustering e inverte quanto fatto dalla operazione di
    "dummying"
    In particolare sfrutta la maniera in cui le tabelle vengono etichettate:
    * se la colonna non contiene un underscore, allora era inizialmente una colonna booleana. Viene eseguita quindi
        l'approssimazione a 0 o a 1 del valore
    * altrimenti si controlla quale delle successive colonne con underscore e con sottostringa il nome della colonna originale
        abbia il valore maggiore. La colonna viene quindi compattata e assegnato il valore-stringa contenuto nella colonna
        con valore maggiore
    al termine viene restituito un dataframe con all'interno i centroidi raccolti dal kmeans
    """

    centroid_df = pd.DataFrame(columns=features)
    if centroid_flag:
        max_iter = len(kmeans.cluster_centers_)
        df_to_invert = kmeans.cluster_centers_
    else:
        max_iter = len(data_frame.index)
        df_to_invert = data_frame.iloc

    for row in range(0, max_iter):
        i = 0
        array_cen = {}
        while i < data_frame.columns.size-1:
            feature_original_name = data_frame.columns.tolist()[i]

            if "_" in data_frame.columns.tolist()[i]:
                feature_original_name = data_frame.columns.tolist()[i][0:feature_original_name.find("_")]
                max_col = i
                i = i+1
                while feature_original_name in data_frame.columns.tolist()[i] and i < data_frame.columns.size-1:
                    if df_to_invert[row, i] > df_to_invert[row, max_col]:
                        max_col = i
                    i = i+1
                array_cen[feature_original_name] = data_frame.columns.tolist()[max_col][(data_frame.columns.tolist()[max_col].find("_")+1):]

            else:
                if df_to_invert[row, i] > 0.5:

                    array_cen[feature_original_name] = 1
                else:
                    array_cen[feature_original_name] = 0
                i = i+1
        if row % 100 == 0:
            print(row)
        centroid_df = centroid_df.append(array_cen, ignore_index=True)
    return centroid_df


def plot_centroids(modified_centroids):
    "Funzione di plotting dei centroidi. La colorazione è eseguita in base alla colonna Labels"
    fig = px.parallel_coordinates(modified_centroids, color="Labels",
                                  color_continuous_scale=px.colors.diverging.Tealrose,
                                  color_continuous_midpoint=2)
    fig.show()


def main():

    """
    Script principale di esecuzione del clustering. Viene letto da linea di comando il percorso del file
    contenente il dataset. Opzionalmente, è possibile specificare:
    - numero di iterate
    - numero di cluster
    """
    try:
        if not path.isfile(argv[1]):
            print("Error: could not find specified CSV dataset.")
            return
    except IndexError:
        print("Error: not enough arguments to start the program" +
              "\n(python clustering.py [file.csv] n_iters n_clusters")
        return

    try:
        if int(argv[2]) > 0:
            iter = arv[2]
        else:
            print("Error: specified non positive number of iterations.")
            return
    except IndexError:
        iter = 10

    try:
        if int(argv[3]) > 0:
            n_clusters = argv[3]
        else:
            print("Error: specified non positive number of clusters.")
            return
    except IndexError:
        n_clusters = 4

    data_frame = pd.read_csv(argv[1])
    (data_frame, features) = refactor_data_frame(data_frame)
    best_cluster = k_means_random_restart(data_frame, iter, n_clusters)
    centroids = dummy_inversion(data_frame, features, best_cluster)
#    centroids.to_csv("centroidi.csv")

    ord_enc = OrdinalEncoder()
    modified_centroids = pd.DataFrame(data=ord_enc.fit_transform(centroids), columns=centroids.columns)
    print(modified_centroids)
    modified_centroids["Labels"] = [i for i in range(1, nclusters+1)]
    plot_centroids(modified_centroids)


main()
