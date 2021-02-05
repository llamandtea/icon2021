import pandas as pd
import datetime as dt

# Caricamento del dataset in memoria
data_frame = pd.read_csv("../hotel_bookings.csv")

# Unione delle serie "children" e "babies"
data_frame = data_frame.assign(minors=data_frame['children'] + data_frame['babies'])
# data_frame.drop("children")
# data_frame.drop("babies")


# Conversione delle date di arrivo in stagioni
# Dizionario da utilizzare per il lookup delle stagioni
seasons = {
    "winter": dt.date(dt.MINYEAR, 12, 21),
    "spring": dt.date(dt.MINYEAR, 3, 20),
    "summer": dt.date(dt.MINYEAR, 6, 21),
    "autumn": dt.date(dt.MINYEAR, 9, 22)
}

# Funzione per convertire la data di arrivo in una riga in una stagione
def convert_to_season(x):
    d = dt.datetime.strptime((x["arrival_date_month"] + " " + x["arrival_date_day_of_month"]), "%B %d")

    if seasons["winter"] <= d < seasons["spring"]:
        return "winter"
    elif seasons["spring"] <= d < seasons["summmer"]:
        return "spring"
    elif seasons["summer"] <= d < seasons["autumn"]:
        return "summer"
    else:
        return "autumn"


# Controllare quale parametro passare a convert_to_season
data_frame = data_frame.assign(season=convert_to_season(data_frame))

# TODO
"""
Normalizzare i valori della colonna 'adr' andando a calcolare la media per ogni
possibile configurazione di <TIPO STANZA, SETTIMANA ARRIVO, ANNO>
Idee per l'implementazione:
- ciclo innestato per il calcolo dei valori della media
- creazione di una look up table
- aggiornamento dei valori sulla tabella
"""

# Scrittura della tabella in un nuovo file
# data_frame.to_csv("..\hotel_bookings_updated.csv")
