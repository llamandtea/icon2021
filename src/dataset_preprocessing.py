"""
Script per il pre-processing del dataset sulle prenotazioni di un hotel
"""
import pandas as pd
import datetime as dt
from math import isnan
from sys import argv
from os import path

# Definizione dell'anno da utilizzare per i calcoli e le date nella look-up table
# L'anno deve essere bisestile per prevenire eventuali errori di parsificazione della data
LEAP_YEAR = int(2016)

# Conversione delle date di arrivo in stagioni
# Dizionario da utilizzare per il lookup delle stagioni
seasons = {
    "winter": dt.date(LEAP_YEAR, 12, 21),
    "spring": dt.date(LEAP_YEAR, 3, 20),
    "summer": dt.date(LEAP_YEAR, 6, 21),
    "autumn": dt.date(LEAP_YEAR, 9, 22)
}


def convert_to_season(x):
    """Funzione per convertire la data di arrivo in una riga in una stagione."""
    d = dt.datetime.strptime(str(x["ArrivalDateYear"]) + " " + x["ArrivalDateMonth"] + " "
                             + str(x["ArrivalDateDayOfMonth"]), "%Y %B %d").date()
    d = d.replace(year=LEAP_YEAR)

    if d < seasons["spring"]:
        return "winter"
    elif d < seasons["summer"]:
        return "spring"
    elif d < seasons["autumn"]:
        return "summer"
    elif d < seasons["winter"]:
        return "autumn"
    else:
        return "winter"


def refactor_lead_time(x):
    """Conversione del valore di 'lead_time'.

    Converte, per le prenotazioni cancellate, il valore di "lead_time" pari al numero di giorni tra la
    prenotazione e la cancellazione
    """
    if x["IsCanceled"] == 1:
        d = (dt.datetime.strptime(str(x["ArrivalDateYear"]) + " " + x["ArrivalDateMonth"] + " "
                                    + str(x["ArrivalDateDayOfMonth"]), "%Y %B %d") - dt.timedelta(x["LeadTime"]))
        return (dt.datetime.strptime(x["ReservationStatusDate"], "%Y-%m-%d") - d).days
    else:
        return x["LeadTime"]


def lookup_adr(x):
    lookup_table = {}
    count_table = {}
    for i, row in x.iterrows():
        key = str(row["ArrivalDateYear"]) + str(row["ArrivalDateWeekNumber"]) + str(row["ReservedRoomType"])\
              + str(row["DistributionChannel"])
        if lookup_table.get(key) is None:
            lookup_table[key] = row["ADR"]
            count_table[key] = 1
        else:
            lookup_table[key] += row["ADR"]
            count_table[key] += 1
    for key in lookup_table:
        # troncamento a due cifre
        lookup_table[key] = "%.2f" % (lookup_table[key]/count_table[key])
    return lookup_table


def refactor_adr(x, table):
    key = str(x["ArrivalDateYear"]) + str(x["ArrivalDateWeekNumber"]) + str(x["ReservedRoomType"])\
          + str(x["DistributionChannel"])
    return table.get(key)


def main(file_path, maintain_original_lead_time=False):
    # Caricamento del dataset in memoria
    if not path.isfile(file_path):
        print("Error: could not find specified CSV dataset")
        return

    data_frame = pd.read_csv(argv[1])

    print("Dropping unused columns...")
    # Prima eliminazione delle colonne non utilizzate/ridondanti
    data_frame = data_frame.drop(
        ["Agent", "Company", "Country", "AssignedRoomType", "RequiredCarParkingSpaces", "ReservationStatus",
         "MarketSegment"],
        axis=1)

    print("Joining Columns...")
    # Unione delle serie "children" e "babies"
    data_frame["Minors"] = data_frame.apply(lambda x: x["Children"]+x["Babies"], axis=1)
    data_frame["Minors"] = data_frame.apply(lambda x: 0 if isnan(x["Minors"]) else int(x["Minors"]), axis=1)
    data_frame = data_frame.drop(["Children", "Babies"], axis=1)

    # Unione delle serie "StaysInWeekendNights" e "StaysInWeekNights"
    data_frame["Staying"] = data_frame.apply(lambda x: x["StaysInWeekendNights"]+x["StaysInWeekNights"], axis=1)
    data_frame["Staying"] = data_frame.apply(lambda x: 0 if isnan(x["Staying"]) else int(x["Staying"]), axis=1)
    data_frame = data_frame.drop(["StaysInWeekendNights", "StaysInWeekNights"], axis=1)

    print("Creating engineered attributes...")
    # normlizzazione del rateo di cancellazione
    data_frame["CancelRate"] = data_frame.apply(lambda x : 0 if
    x["PreviousCancellations"] + x["PreviousBookingsNotCanceled"] == 0 else
    x["PreviousCancellations"]/(x["PreviousCancellations"] + x["PreviousBookingsNotCanceled"]), axis=1)
    data_frame = data_frame.drop(["PreviousCancellations", "PreviousBookingsNotCanceled"], axis=1)

    # Conversione della feature sui giorni in lista d'attesa in feature booleana
    data_frame["WasInWaitingList"] = data_frame.apply(lambda x: 1 if x["DaysInWaitingList"] > 0 else 0, axis=1)
    data_frame = data_frame.drop("DaysInWaitingList", axis=1)

    table = lookup_adr(data_frame)
    data_frame["Season"] = data_frame.apply(convert_to_season, axis=1)

    if maintain_original_lead_time:
        data_frame["OriginalLeadTime"] = data_frame["LeadTime"]

    data_frame["LeadTime"] = data_frame.apply(refactor_lead_time, axis=1)

    data_frame = data_frame.drop(["ReservationStatusDate", "ArrivalDateMonth", "ArrivalDateDayOfMonth"], axis=1)

    """
    Normalizzazione dei valori dell'ADR come media per possibile configurazione di 
    <ANNO ARRIVO, SETTIMANA ARRIVO, STANZA RISERVATA, CANALE DI DISTRIBUZIONE>

    args prende una tupla di argomenti in input, dato che il passaggio di data_frame è
    implicito, viene avvalorato solo con la tupla a singolo valore (table, )
    """
    data_frame["ADR"] = data_frame.apply(refactor_adr, axis=1, args=(table, ))
    data_frame = data_frame.drop(["ReservedRoomType", "ArrivalDateWeekNumber", "ArrivalDateYear"], axis=1)



    # Scrittura della tabella in un nuovo file
    # nb. lasciare r preposto al path per l'utilizzo di una stringa raw

    data_frame.to_csv((path.dirname(argv[1]) + "//" + "processed_" + path.basename(argv[1])), index=False)
    print("Done.")


main(argv[1], True)

