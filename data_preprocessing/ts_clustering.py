import os
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tslearn.clustering import TimeSeriesKMeans
from tslearn.barycenters import dtw_barycenter_averaging

seed = 42

__file__ = "your_path/ts_clustering.py"

# load data
rifiuti_puglia = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "../data", "rifiuti_puglia_scaled.csv")
)
features_comuni_global = np.load(
    os.path.join(os.path.dirname(__file__), "../data", "features_comuni_global.npy"),
    allow_pickle=True,
).item()
confini = gpd.read_file(
    os.path.join(os.path.dirname(__file__), "../data", "confini_comuni_puglia.shp")
)

# prepare data
grouped_data = (
    rifiuti_puglia.groupby("comune")["tot_rsu"].apply(list).reset_index(name="rifiuti")
)
time_series = np.array(grouped_data["rifiuti"].values.tolist())


# time series clustering with kmeans dtw
pca = PCA(n_components=3)
time_series_pca = pca.fit_transform(time_series)
cluster_count = 4
km_pca = TimeSeriesKMeans(
    n_clusters=cluster_count, metric="dtw", verbose=False, random_state=seed
)
labels_pca = km_pca.fit_predict(time_series)


# plot clusters
plot_count = math.ceil(math.sqrt(cluster_count))

fig, axs = plt.subplots(plot_count, plot_count, figsize=(25, 25))
fig.suptitle("Clusters")
row_i = 0
column_j = 0
for label in set(labels_pca):
    cluster = []
    for i in range(len(labels_pca)):
        if labels_pca[i] == label:
            axs[row_i, column_j].plot(time_series[i], c="gray", alpha=0.4)
            cluster.append(time_series[i])
    if len(cluster) > 0:
        axs[row_i, column_j].plot(dtw_barycenter_averaging(np.vstack(cluster)), c="red")
    axs[row_i, column_j].set_title("Cluster " + str(row_i * plot_count + column_j))
    column_j += 1
    if column_j % plot_count == 0:
        row_i += 1
        column_j = 0

plt.show()

# apply clusters
names_for_labels = [f"Cluster {label}" for label in labels_pca]
ts_clusters = (
    pd.DataFrame(
        zip(grouped_data["comune"], names_for_labels),
        columns=["comune", "cluster"],
    )
    .sort_values("cluster")
    .set_index("comune")
)

for comune in rifiuti_puglia["comune"].unique():
    ts_clusters.loc[comune, "lat"] = features_comuni_global[comune]["latitudine"]
    ts_clusters.loc[comune, "lon"] = features_comuni_global[comune]["longitudine"]
    ts_clusters.loc[comune, "area"] = features_comuni_global[comune]["area"]
    ts_clusters.loc[comune, "popolazione"] = features_comuni_global[comune][
        "popolazione"
    ]


ts_clusters.reset_index(inplace=True)
gdf = pd.merge(ts_clusters, confini, on="comune")
gdf = gpd.GeoDataFrame(gdf)
gdf.to_file(os.path.join(os.path.dirname(__file__), "../data", "ts_clusters_PCA.shp"))

# convert geometry in points considering the centroid
gdf["geometry"] = gdf["geometry"].centroid
gdf.to_file(os.path.join(os.path.dirname(__file__), "../data", "ts_clusters_PCA.shp"))
