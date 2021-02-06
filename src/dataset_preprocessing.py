"""
Script per il pre-processing del dataset sulle prenotazioni di un hotel
"""
import pandas as pd
import datetime as dt
from math import isnan

# Conversione delle date di arrivo in stagioni
# Dizionario da utilizzare per il lookup delle stagioni
seasons = {
    "winter": dt.date(1944, 12, 21),
    "spring": dt.date(1944, 3, 20),
    "summer": dt.date(1944, 6, 21),
    "autumn": dt.date(1944, 9, 22)
}


def convert_to_season(x):
    """Funzione per convertire la data di arrivo in una riga in una stagione."""
    d = dt.datetime.strptime(str(x["arrival_date_year"]) + " " + x["arrival_date_month"] + " "
                             + str(x["arrival_date_day_of_month"]), "%Y %B %d").date()
    d = d.replace(year=1944)

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
    if x["is_canceled"] == 1:
        d = (dt.datetime.strptime(str(x["arrival_date_year"]) + " " + x["arrival_date_month"] + " "
                                    + str(x["arrival_date_day_of_month"]), "%Y %B %d") - dt.timedelta(x["lead_time"]))
        return (dt.datetime.strptime(x["reservation_status_date"], "%Y-%m-%d") - d).days
    else:
        return x["lead_time"]


def lookup_adr(x):
    lookup_table = {}
    count_table = {}
    for i, row in x.iterrows():
        key = str(row["arrival_date_year"]) + str(row["arrival_date_week_number"]) + str(row["reserved_room_type"])\
              + str(row["distribution_channel"])
        if lookup_table.get(key) is None:
            lookup_table[key] = row["adr"]
            count_table[key] = 1
        else:
            lookup_table[key] += row["adr"]
            count_table[key] += 1
    for key in lookup_table:
        # troncamento a due cifre
        lookup_table[key] = "%.2f" % (lookup_table[key]/count_table[key])
    return lookup_table


def refactor_adr(x, table):
    key = str(x["arrival_date_year"]) + str(x["arrival_date_week_number"]) + str(x["reserved_room_type"])\
          + str(x["distribution_channel"])
    return table.get(key)


def main():
    # Caricamento del dataset in memoria
    data_frame = pd.read_csv("../hotel_bookings.csv")

    # TODO: pulizia del dataset
    # Prima eliminazione delle colonne non utilizzate
    data_frame = data_frame.drop(
        ["agent", "company", "country", "assigned_room_type", "required_car_parking_spaces", "reservation_status"],
        axis=1)

    # Unione delle serie "children" e "babies"
    data_frame["minors"] = data_frame.apply(lambda x: x["children"]+x["babies"], axis=1)
    data_frame["minors"] = data_frame.apply(lambda x: 0 if isnan(x["minors"]) else int(x["minors"]), axis=1)
    data_frame = data_frame.drop(["children", "babies"], axis=1)
    
    #normlizzazione del rateo di cancellazione

    data_frame["cancel_rate"] = data_frame.apply(lambda x : 0 if
    x["previous_cancellations"] + x["previous_bookings_not_canceled"]==0 else
    x["previous_cancellations"]/(x["previous_cancellations"] + x["previous_bookings_not_canceled"]), axis = 1)
    data_frame = data_frame.drop(["previous_cancellations", "previous_bookings_not_canceled"], axis=1)
    # Conversione della feature sui giorni in lista d'attesa in feature booleana
    data_frame["was_in_waiting_list"] = data_frame.apply(lambda x: 1 if x["days_in_waiting_list"] > 0 else 0, axis=1)
    data_frame = data_frame.drop("days_in_waiting_list", axis=1)

    table = lookup_adr(data_frame)
    data_frame["season"] = data_frame.apply(convert_to_season, axis=1)
    data_frame["lead_time"] = data_frame.apply(refactor_lead_time, axis=1)
    data_frame = data_frame.drop(["reservation_status_date", "arrival_date_month", "arrival_date_day_of_month"], axis=1)

    """
    Normalizzazione dei valori dell'ADR come media per possibile configurazione di 
    <ANNO ARRIVO, SETTIMANA ARRIVO, STANZA RISERVATA, CANALE DI DISTRIBUZIONE>

    args prende una tupla di argomenti in input, dato che il passaggio di data_frame è
    implicito, viene avvalorato solo con la tupla a singolo valore (table, )
    """
    data_frame["adr"] = data_frame.apply(refactor_adr, axis=1, args=(table, ))
    data_frame = data_frame.drop(["reserved_room_type", "arrival_date_week_number", "arrival_date_year"], axis=1)

    # Scrittura della tabella in un nuovo file
    # nb. lasciare r preposto al path per l'utilizzo di una stringa raw
    data_frame.to_csv(r"..\hotel_bookings_updated.csv", index=False)


main()
