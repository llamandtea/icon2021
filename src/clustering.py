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


iter = 10
data_frame = pd.read_csv("../datasets/processed_resort_hotels.csv")
"""
K = [["Direct"], ["Corporate"], ["TA/TO"], ["Undefined"]]
enc.fit(K)

#X = data_frame["DistributionChannel"].tolist()
#X = X.reshape(1, X.shape[0])
#X = np.array(["Direct", "TA/TO", "Corporate"]).reshape(3, -1)

X = data_frame["DistributionChannel"].to_numpy()
X = X.reshape(X.shape[0], -1)
Y = enc.transform(X)
print(K)
print(X)
print(Y)
"""

data_frame = data_frame.assign(Staying = data_frame["StaysInWeekendNights"] + data_frame["StaysInWeekNights"])

ord_enc = OrdinalEncoder()
ord_enc = ord_enc.fit(data_frame)

X_adr = pd.cut(data_frame["ADR"], 3) #discretizza una colonna in n=3 intervalli (bins)

X_lead = pd.cut(data_frame["LeadTime"], [0, 7, 30, 90, 180, 365, data_frame["LeadTime"].max()], include_lowest=True)
X_stays_st = pd.cut(data_frame["Staying"], [0, 3, 7, 15, data_frame["Staying"].max()], include_lowest=True)
#X_stays_wn = pd.cut(data_frame["StaysInWeekNights"], 4)
X_booking_ch = pd.cut(data_frame["BookingChanges"], [0, 5, 10, data_frame["BookingChanges"].max()], include_lowest=True)
X_adults = pd.cut(data_frame["Adults"], [0,1,data_frame["Adults"].max()], include_lowest=True)
                  # 3 [0, 1, data_frame["Adults"].max()])
X_cancel_rate = pd.cut(data_frame["CancelRate"], 2)
X_minors = pd.cut(data_frame["Minors"], [0, 2, data_frame["Minors"].max()], include_lowest=True)

data_frame = data_frame.drop(["ADR", "LeadTime", "StaysInWeekendNights", "StaysInWeekNights", "BookingChanges", "Adults", "CancelRate", "Minors"], axis=1)
data_frame = data_frame.drop(["MarketSegment"], axis = 1)
#data_frame = pd.concat([data_frame, X_adr, X_lead, X_stays_we, X_stays_wn, X_booking_ch], ignore_index=True, axis=1)


data_frame = data_frame.assign(ADR=X_adr)
data_frame = data_frame.assign(LeadTime=X_lead)
#data_frame = data_frame.assign(StaysInWeekendNights=X_stays_we)
#data_frame = data_frame.assign(StaysInWeekNights=X_stays_wn)
data_frame = data_frame.assign(BookingChanges=X_booking_ch)
data_frame = data_frame.assign(Adults=X_adults)
data_frame = data_frame.assign(CancelRate=X_cancel_rate)
data_frame = data_frame.assign(Minors=X_minors)
data_frame = data_frame.assign(Staying = X_stays_st)

features = data_frame.columns.tolist()


data_frame = pd.get_dummies(data_frame)


"""
kf = KFold(n_splits=2, shuffle=True)
df = data_frame.to_numpy()
for startingIndex, endingIndex in kf.split(data_frame):
    data_frame = data_frame.iloc[startingIndex, :]
    break;
"""


best_cluster = cluster.KMeans(n_clusters=4, random_state=np.random.randint(1, 40000)).fit(data_frame) #clusterizzazione effettiva
for i in range(0, iter):
    current_cluster = cluster.KMeans(n_clusters=4, random_state=np.random.randint(1, 40000)).fit(data_frame) #clusterizzazione effettiva
    print(current_cluster.inertia_)
    if best_cluster.inertia_ > current_cluster.inertia_ :
        best_cluster=current_cluster

print("Best inertia: " + str(best_cluster.inertia_))

kmeans = best_cluster
data_frame["Labels"] = kmeans.labels_
#fig.show()
centroid_df = pd.DataFrame(columns=features)


print(data_frame.columns)
for centroid in range (0, len(kmeans.cluster_centers_)):
    i = 0
    array_cen = {}
    while i < data_frame.columns.size-1:
        feature_original_name = data_frame.columns.tolist()[i]

        if "_" in data_frame.columns.tolist()[i]:
            feature_original_name = data_frame.columns.tolist()[i][0:feature_original_name.find("_")]
            max_col = i
            i = i+1
            while feature_original_name in data_frame.columns.tolist()[i] and i < data_frame.columns.size-1:
                if kmeans.cluster_centers_[centroid, i] > kmeans.cluster_centers_[centroid, max_col]:
                    max_col = i
                i = i+1
            array_cen[feature_original_name] = data_frame.columns.tolist()[max_col][(data_frame.columns.tolist()[max_col].find("_")+1) : ]

        else:
            if kmeans.cluster_centers_[centroid, i] > 0.5:

                array_cen[feature_original_name] = 1
            else:
                array_cen[feature_original_name] = 0
            i = i+1
    centroid_df = centroid_df.append(array_cen, ignore_index=True)



centroid_df.to_csv("centroidi.csv")

"""
for col in centroid_df.columns:
    modified_centroids[col] = ord_enc.fit_transform(centroid_df[[col]])
"""

modified_centroids = pd.DataFrame(data=ord_enc.transform(centroid_df), columns=centroid_df.columns)


print(modified_centroids)

modified_centroids["Labels"] = [1,2,3,4]

fig = px.parallel_coordinates(modified_centroids, color="Labels",
                              color_continuous_scale=px.colors.diverging.Tealrose,
                              color_continuous_midpoint=2)
fig.show()

#fig = px.parallel_coordinates(cenrtoid_df, color="Class")
"""
fig = pd.plotting.parallel_coordinates(centroid_df, class_column = 'Class', color=('#556270', '#4ECDC4', '#C7F464'))
fig.show()
#da finire
"""
#print(Y)
#Z = np.asarray(Y)
#print(type(Z))
