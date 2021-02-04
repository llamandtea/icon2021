import pandas as pd
from datetime import datetime

# Caricamento del dataset in memoria
data_frame = pd.read_csv("..\hotel_bookings.csv")

# Unione delle serie "children" e "babies"
data_frame["minors"] = data_frame["children"] + data_frame["babies"]
data_frame.drop(["children", "babies"])

# TODO
"""
Normalizzare i valori della colonna 'adr' andando a calcolare la media per ogni
possibile configurazione di <TIPO STANZA, SETTIMANA ARRIVO, ANNO>
Idee per l'implementazione:
- ciclo innestato per il calcolo dei valori della media
- creazione di una look up table
- aggiornamento dei valori sulla tabella
"""

data_frame.to_csv("..\hotel_bookings_updated.csv")