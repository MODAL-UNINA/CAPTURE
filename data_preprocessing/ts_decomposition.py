import os
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

__file__ = "your_path/ts_decomposition.py"

# load data
rifiuti_puglia = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "../data", "rifiuti_puglia.csv")
)
idx_cluster = np.load(
    os.path.join(os.path.dirname(__file__), "../data", "idx_cluster.npy"),  
    allow_pickle=True).item()

# scale data
scaler_dict = {k: {"scaler": {}, "scaled_data": {}} for k in idx_cluster.keys()}
for idx_clust in idx_cluster.keys():
    scaler = MinMaxScaler()
    comuni_data = rifiuti_puglia[rifiuti_puglia["comune"].isin(idx_cluster[idx_clust])]
    scaler.fit(comuni_data[["tot_rsu"]])
    scaler_dict[idx_clust]["scaler"] = scaler
    comuni_data["tot_rsu"] = scaler.transform(comuni_data[["tot_rsu"]])
    scaler_dict[idx_clust]["scaled_data"] = comuni_data

# decompose time series
decomposed_clusters = {
    k: pd.DataFrame(
        columns=["mese", "comune", "tot_rsu", "trend", "seasonal", "residual"]
    )
    for k in idx_cluster.keys()
}

for idx_clust in idx_cluster.keys():
    for comune in tqdm(idx_cluster[idx_clust]):
        comune_data = scaler_dict[idx_clust]["scaled_data"][
            scaler_dict[idx_clust]["scaled_data"]["comune"] == comune
        ]
        comune_data = comune_data[["mese", "tot_rsu"]]
        comune_data = comune_data.set_index("mese")
        comune_data.index = pd.to_datetime(comune_data.index)
        comune_data = comune_data.resample("M").sum()
        comune_data = comune_data.interpolate(method="linear")

        result = STL(comune_data, period=12, seasonal=15).fit()
        comune_data["trend"] = result.trend
        comune_data["seasonal"] = result.seasonal
        comune_data["residual"] = result.resid
        comune_data["comune"] = comune
        decomposed_clusters[idx_clust] = pd.concat(
            [decomposed_clusters[idx_clust], comune_data.reset_index()]
        )

    decomposed_clusters[idx_clust].reset_index(drop=True, inplace=True)

# create sliding windows
def create_windows(df, window, horizon, nodes_dict, col: list, step_window: int = 1):
    X = []
    Y = []
    id_comune = []
    comuni = df["comune"].unique()
    for comune in comuni:
        comune_data = df[df["comune"] == comune]
        comune_data = comune_data[col].values
        for i in range(0, len(comune_data) - window - horizon + 1, step_window):
            x = comune_data[i : i + window]
            y = comune_data[i + window : i + window + horizon]

            X.append(x)
            Y.append(y)
            id_comune.append(nodes_dict[comune])

    return np.array(X), np.array(Y), np.array(id_comune)

# set parameters
window = 24
step_window = 1
horizon = 18

# nodes dictionary
indexes = [i for i in range(0, len(rifiuti_puglia["comune"].unique()))]
zip_iterator = zip(rifiuti_puglia["comune"].unique(), indexes)
nodes_dict = dict(zip_iterator)

# train and test split
df_train = {k: pd.DataFrame() for k in idx_cluster.keys()}
df_test = {k: pd.DataFrame() for k in idx_cluster.keys()}
for idx_clust in idx_cluster.keys():
    all_mesi_test = (
        decomposed_clusters[idx_clust]
        .sort_values("mese")["mese"]
        .unique()[-(window + horizon) :]
    )
    mesi_test = all_mesi_test[-horizon:]
    df_test[idx_clust] = decomposed_clusters[idx_clust][
        decomposed_clusters[idx_clust]["mese"].isin(all_mesi_test)
    ]
    df_train[idx_clust] = decomposed_clusters[idx_clust][
        decomposed_clusters[idx_clust]["mese"] >= "2014-01-01"
    ]
    df_train[idx_clust] = df_train[idx_clust][
        ~df_train[idx_clust]["mese"].isin(mesi_test)
    ]

col = ["tot_rsu", "trend", "seasonal", "residual"]
X_train = {k: [] for k in idx_cluster.keys()}
y_train = {k: [] for k in idx_cluster.keys()}
id_comune_train = {k: [] for k in idx_cluster.keys()}
X_test = {k: [] for k in idx_cluster.keys()}
y_test = {k: [] for k in idx_cluster.keys()}
id_comune_test = {k: [] for k in idx_cluster.keys()}

for idx_clust in idx_cluster.keys():
    X_train[idx_clust], y_train[idx_clust], id_comune_train[idx_clust] = create_windows(
        df_train[idx_clust], window, horizon, nodes_dict, col, step_window
    )
    X_test[idx_clust], y_test[idx_clust], id_comune_test[idx_clust] = create_windows(
        df_test[idx_clust], window, horizon, nodes_dict, col, step_window
    )

# create torch loader with batch size and shuffle
batch_size = 256

train_data = {
    k: TensorDataset(
        torch.tensor(X_train[k], dtype=torch.float32),
        torch.tensor(y_train[k], dtype=torch.float32),
        torch.tensor(id_comune_train[k], dtype=torch.long),
    )
    for k in idx_cluster.keys()
}
train_loader = {
    k: DataLoader(v, shuffle=True, batch_size=batch_size) for k, v in train_data.items()
}

test_data = {
    k: TensorDataset(
        torch.tensor(X_test[k], dtype=torch.float32),
        torch.tensor(y_test[k], dtype=torch.float32),
        torch.tensor(id_comune_test[k], dtype=torch.long),
    )
    for k in idx_cluster.keys()
}
test_loader = {
    k: DataLoader(v, shuffle=False, batch_size=batch_size) for k, v in test_data.items()
}

# save data
np.save(
    os.path.join(os.path.dirname(__file__), "../data", "train_loader.npy"), train_loader
)
np.save(
    os.path.join(os.path.dirname(__file__), "../data", "test_loader.npy"), test_loader
)
np.save(
    os.path.join(os.path.dirname(__file__), "../data", "scaler_dict.npy"), scaler_dict
)