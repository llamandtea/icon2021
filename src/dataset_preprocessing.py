import pandas as pd
import datetime as dt

# Caricamento del dataset in memoria
data_frame = pd.read_csv("../hotel_bookings.csv")

# Unione delle serie "children" e "babies"
data_frame = data_frame.assign(minors=data_frame['children'] + data_frame['babies'])

# Conversione delle date di arrivo in stagioni
# Dizionario da utilizzare per il lookup delle stagioni
seasons = {
    "winter": dt.date(1944, 12, 21),
    "spring": dt.date(1944, 3, 20),
    "summer": dt.date(1944, 6, 21),
    "autumn": dt.date(1944, 9, 22)
}


# Funzione per convertire la data di arrivo in una riga in una stagione
def convert_to_season(x):
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
        key = str(row["arrival_date_year"]) + str(row["arrival_date_week_number"]) + str(row["reserved_room_type"]) + str(row["distribution_channel"])
        if lookup_table.get(key) is None:
            lookup_table[key] = row["adr"]
            count_table[key] = 1
        else:
            lookup_table[key] += row["adr"]
            count_table[key] +=1
    for key in lookup_table:
        #TODO: troncare a due cifre
        lookup_table[key] /= count_table[key]
    return lookup_table

table = lookup_adr(data_frame)
print(table)

#def refactor_adr(x):



#data_frame["adr"] = data_frame.apply(refactor_adr, axis=1)
#data_frame["was_in_waiting_list"] = data_frame.apply(lambda x: x["days_in_waiting_list"] > 0, axis=1)
#data_frame["season"] = data_frame.apply(convert_to_season, axis=1)
#data_frame["lead_time"] = data_frame.apply(refactor_lead_time, axis=1)

# Cancellazione delle colonne non utilizzate

# data_frame.drop(["children", "babies", "days_in_waiting_list"], axis=1)
# data_frame.drop(["arrival_date_week_number", "arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"], axis=1)
# data_frame.drop(["agent", "company"], axis=1)
#data_frame.drop(["reserved_room_type", "assigned_room_type", "required_car_parking"], axis=1)
# data_frame.drop(["reservation_status", "reservation_status_date"], axis = 1)

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
