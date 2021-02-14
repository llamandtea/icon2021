"""
Importante: prendere nota dei commenti alla funzione main() prima di avviare lo script

Per quanto concerne la funzione di plot per il metodo del gomito, può essere eseguita da riga di comando o
decommentando l'ultima riga dello script

Autori: Dell'Olio Domenico, Delvecchio Giovanni Pio, Disabato Raffaele, Lamantea Giuseppe
"""
import numpy as np
import pandas as pd
from sklearn import cluster
from matplotlib import pyplot as plt
from os import path
from sys import argv


def refactor_data_frame(data_frame):
    """
    Funzione che, preso in input il DataFrame contenente i dati da clusterizzare
    converte tutte le feature continue prima in split e poi in feature indicatrici
    restituisce inoltre il dataframe modificato e il nome delle colonne prima dell'applicazione del dummy
    """

    """
    Discretizzazione delle colonne continue in intervalli n-ari. Per ogni colonna sono stati scelti degli estremi
    significativi quando possibile.
    """
    x_adr = pd.cut(data_frame["ADR"], 3)
    x_lead = pd.cut(data_frame["LeadTime"], [0, 7, 30, 90, 180, 365, data_frame["LeadTime"].max()], include_lowest=True)
    x_stays = pd.cut(data_frame["Staying"], [0, 3, 7, 15, data_frame["Staying"].max()], include_lowest=True)
    x_booking_ch = pd.cut(data_frame["BookingChanges"], [0, 5, 10, data_frame["BookingChanges"].max()],
                          include_lowest=True)
    x_adults = pd.cut(data_frame["Adults"], [0,1,data_frame["Adults"].max()], include_lowest=True)
    x_cancel_rate = pd.cut(data_frame["CancelRate"], 2)
    x_minors = pd.cut(data_frame["Minors"], [0, 2, data_frame["Minors"].max()], include_lowest=True)

    # Eliminazione delle colonne convertite dal dataframe originale
    data_frame = data_frame.drop(["ADR",
                                  "LeadTime",
                                  "Staying",
                                  "BookingChanges",
                                  "Adults",
                                  "CancelRate",
                                  "Minors"], axis=1)

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


def k_means_random_restart(data_frame, iterations, nclusters):
    """
    Funzione che effettua il clustering attraverso quanto messo a disposizione dalla libreria sklearn per ottenere
    nclusters cluster. Iter indica, invece, il numero random restart da rieseguire per ottenere l'ottimo locale.
    Non viene utilizzato il parametro n_init della funzione di sklearn perchè si vuole avere un feedback sull'andamento
    delle iterazioni
    """
    print("Starting clustering...")

    # Inizializzazione del miglior cluster
    best_cluster = cluster.KMeans(n_clusters=nclusters, random_state=np.random.randint(1, 40000)).fit(data_frame)

    for i in range(0, iterations):
        current_cluster = cluster.KMeans(n_clusters=nclusters, n_init=1).fit(data_frame)
        if best_cluster.inertia_ > current_cluster.inertia_:
            best_cluster = current_cluster
        if (i + 1) % 10 == 0:
            print("Iteration number " + str(i + 1))

    print("Best inertia: " + str(best_cluster.inertia_))
    return best_cluster


def dummy_inversion(data_frame, features, kmeans=None):
    """
    Preso in input il dataframe, le features e e quanto ottenuto dal clustering e inverte quanto fatto dalla operazione di
    "dummying". Alternativamente, se kmeans non viene passato, viene fatta l'inversione del dataframe
    In particolare sfrutta la maniera in cui le tabelle vengono etichettate:
    * se la colonna non contiene un underscore, allora era inizialmente una colonna booleana. Viene eseguita quindi
        l'approssimazione a 0 o a 1 del valore

    * altrimenti si controlla quale delle successive colonne con underscore e con sottostringa il nome della colonna originale
        abbia il valore maggiore. La colonna viene quindi compattata e assegnato il valore-stringa contenuto nella colonna
        con valore maggiore

    al termine viene restituito un dataframe con all'interno i centroidi raccolti dal kmeans
    """

    centroid_df = pd.DataFrame(columns=features)
    if kmeans is not None:
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
                array_cen[feature_original_name] = data_frame.columns.tolist()[max_col][(data_frame.columns.tolist()
                                                                                         [max_col].find("_")+1):]

            else:
                if df_to_invert[row, i] > 0.5:

                    array_cen[feature_original_name] = 1
                else:
                    array_cen[feature_original_name] = 0
                i = i+1
        if row % 100 == 0 and kmeans is None:
            print(row)
        centroid_df = centroid_df.append(array_cen, ignore_index=True)
    return centroid_df


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
            iterations = int(argv[2])
        else:
            print("Error: specified non positive number of iterations.")
            return
    except IndexError:
        iterations = 10

    try:
        if int(argv[3]) > 0:
            n_clusters = int(argv[3])
        else:
            print("Error: specified non positive number of clusters.")
            return
    except IndexError:
        n_clusters = 4

    data_frame = pd.read_csv(argv[1])
    (data_frame, features) = refactor_data_frame(data_frame)
    best_cluster = k_means_random_restart(data_frame, iterations, n_clusters)
    centroids = dummy_inversion(data_frame, features, kmeans=best_cluster)

    # ai centroidi si vuole aggiungere il numero di esempi che raggruppano
    labels, count = np.unique(best_cluster.labels_, return_counts=True)
    centroids = centroids.assign(N_Examples=count)
    centroids.to_csv((path.dirname(argv[1]) + "//" + str(n_clusters) + " centroidi_" + path.basename(argv[1])),
                     index=False)


def k_elbow_plot(fpath, max_k=10):
    """
    Funzione per plottare il grafico "a gomito" che mostra il rapporto tra SSE del modello di clustering
    e il numero di cluster scelti. L'utilità sta nel poter selezionare il numero di k più appropriato
    in base all'ultimo valore k che comporta una buona diminuzione dell'errore ("punta del gomito")
    :param fpath: percorso del dataset processato (vedasi descrizione di argv[1] in cima allo script)
    :param max_k: numero massimo di k che si vuole utilizzare per produrre il grafo
    """
    if not path.isfile(fpath):
        print("Error: could not find specified CSV dataset.")
        return
    if max_k <= 0:
        print("Error: k must be a positive integer.")
        return

    data, features = refactor_data_frame(pd.read_csv(fpath))
    errors = []
    for k in range(1, max_k+1):
        kmeans = cluster.KMeans(n_clusters=k)
        kmeans.fit(data)
        errors.append(kmeans.inertia_)
        print("DONE WITH K=" + str(k))
    plt.figure(figsize=(16, 8))
    plt.plot(range(1, max_k+1), errors, 'bo-')
    plt.xlabel('#Clusters (K)')
    plt.ylabel('Errore (SSE)')
    plt.title("Rapporto parametro K/errore del dataset " + path.basename(fpath))
    plt.show()


main()
# k_elbow_plot(fpath="../datasets/processed_city_hotel.csv", max_k=15)
