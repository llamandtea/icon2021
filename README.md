# icon2021
Repository per il progetto di Ingegneria della Conoscenza 2020/2021 (Informatica, Bari)

## Impostare il progetto

- Clonare il progetto
```
git clone https://github.com/llamandtea/icon2021.git
```

- Creare il virtual env
```
cd icon2021
python -m venv icon2021
```

- Attivare il virtual env
```
(linux o git bash su win)
source /src/icon2021/bin/activate
```

- Installare le dipendenze
```
pip install -r requirements.txt
```

## Esecuzione degli script

### Preprocessing del Dataset

```
dataset_preprocessing.py ./datasets/resort_hotel.csv
dataset_preprocessing.py ./datasets/city_hotel.csv
```

### Apprendimento degli alberi

```
tree_learning.py ./datasets/tree_resort_hotel.csv [numero_folds]
tree_learning.py ./datasets/tree_city_hotel.csv [numero_folds]
```

### Appprendimento dei cluster

```
clustering.py ./datasets/network_cluster_city_hotel.csv [numero_iterazioni] [numero_fold]
clustering.py ./datasets/network_cluster_resort_hotel.csv [numero_iterazioni] [numero_fold]
```
