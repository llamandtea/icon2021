"""
Importante: prendere nota dei commenti alla funzione main() prima di avviare lo script

Per quanto concerne la funzione di plot per il metodo del gomito, può essere eseguita da riga di comando o
decommentando l'ultima riga dello script

Autori: Dell'Olio Domenico, Delvecchio Giovanni Pio, Disabato Raffaele, Lamantea Giuseppe
"""
import numpy as np
import pandas as pd
from kmodes.kmodes import KModes
from matplotlib import pyplot as plt
from os import path
from sys import argv
import pickle

def refactor_data_frame(data_frame):
    """
    Funzione che, preso in input il DataFrame contenente i dati da clusterizzare
    converte tutte le feature continue prima in split e poi in feature indicatrici
    restituisce inoltre il dataframe modificato
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
    x_special_req = pd.cut(data_frame["TotalOfSpecialRequests"], [0,1,3, data_frame["TotalOfSpecialRequests"].max()],
                           include_lowest=True)

    # Eliminazione delle colonne convertite dal dataframe originale
    data_frame = data_frame.drop(["ADR",
                                  "LeadTime",
                                  "Staying",
                                  "BookingChanges",
                                  "Adults",
                                  "CancelRate",
                                  "Minors",
                                  "TotalOfSpecialRequests"], axis=1)

    # Sostituzione delle colonne eliminate con quelle discretizzate
    data_frame = data_frame.assign(ADR=x_adr)
    data_frame = data_frame.assign(LeadTime=x_lead)
    data_frame = data_frame.assign(BookingChanges=x_booking_ch)
    data_frame = data_frame.assign(Adults=x_adults)
    data_frame = data_frame.assign(CancelRate=x_cancel_rate)
    data_frame = data_frame.assign(Minors=x_minors)
    data_frame = data_frame.assign(Staying=x_stays)
    data_frame = data_frame.assign(TotalOfSpecialRequests=x_special_req)

    return data_frame


def k_means_random_restart(data_frame, iterations, nclusters):
    """
    Funzione che effettua il clustering attraverso quanto messo a disposizione dalla libreria sklearn per ottenere
    nclusters cluster. Iter indica, invece, il numero random restart da rieseguire per ottenere l'ottimo locale.
    """
    print("Starting clustering...")

    # Inizializzazione del miglior cluster
    best_cluster = KModes(n_clusters=nclusters, n_init=iterations, init="random").fit(data_frame)

#    for i in range(0, iterations):
#        current_cluster = KModes(n_clusters=nclusters, n_init=1, max_iter=300).fit(data_frame)
#        if best_cluster.cost_ > current_cluster.cost_:
#            best_cluster = current_cluster
#        if (i + 1) % 10 == 0:
#            print("Iteration number " + str(i + 1))

    print("Best cost: " + str(best_cluster.cost_))
    return best_cluster


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
    data_frame = refactor_data_frame(data_frame)
    best_cluster = k_means_random_restart(data_frame, iterations, n_clusters)
    centroids = pd.DataFrame(best_cluster.cluster_centroids_, columns=data_frame.columns.tolist())

    # ai centroidi si vuole aggiungere il numero di esempi che raggruppano
    labels, count = np.unique(best_cluster.labels_, return_counts=True)
    centroids = centroids.assign(N_Examples=count)
    centroids.to_csv((path.dirname(argv[1]) + "//" + str(n_clusters) + " centroidi_" + path.basename(argv[1])),
                     index=False)

    save_path = "..//models//" + path.basename(argv[1].replace(".csv", "_best.cluster"))
    with open( save_path, "wb") as out_cluster:
        pickle.dump(best_cluster, out_cluster)


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

    data = refactor_data_frame(pd.read_csv(fpath))
    errors = []
    for k in range(1, max_k+1):
        kmodes = KModes(n_clusters=k, random_state=42, n_init=1, init="random")
        kmodes.fit(data)
        errors.append(kmodes.cost_)
        print("DONE WITH K=" + str(k))
    plt.figure(figsize=(16, 8))
    plt.plot(range(1, max_k+1), errors, 'bo-')
    plt.xlabel('#Clusters (K)')
    plt.ylabel('Errore (0/1)')
    plt.title("Rapporto parametro K/errore del dataset " + path.basename(fpath))
    plt.show()


main()
# k_elbow_plot(fpath="../datasets/processed_city_hotel.csv", max_k=15)
