#%%
import os
import torch
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange
from IPython.display import display
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils.models import model_complete0, model_complete1, model_complete2, model_complete3, my_loss

import warnings
warnings.filterwarnings("ignore")

# set seed
import random
seed = 42

# set device
device_number = 0
device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")

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

# train function
def train(train_loader, test_loader, scaler, model_dict, graph_inputs):

    model = model_dict["model"]
    lr = model_dict.get("lr", 0.001)
    patience = model_dict.get("patience", 100)
    epochs = model_dict.get("epochs", 1500)
    name = model_dict.get("name", "model")
    wd = model_dict.get("wd", 1e-5)
    model = model.to(device)

    criterion = my_loss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    train_dict = {}
    val_dict = {}
    measures_train = {}
    measures_val = {}

    wait = 0
    val_best = np.inf
    best_epoch = -1
    best_state = model.state_dict()

    for epoch in (pbar := trange(epochs, desc=name)):
        model.train()
        running_loss = {}
        for x_ts, y, idx in train_loader:
            x_ts = torch.tensor(x_ts).to(device)
            target = torch.tensor(y).to(device)
            target = target.squeeze()
            idx = torch.tensor(idx).to(device)

            optimizer.zero_grad()
            predictions = model(x_ts, idx, **graph_inputs)
            loss_dict = criterion(predictions, target[:, :, 0])
            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()
            for k, v in loss_dict.items():
                running_loss.setdefault(k, []).append(v.item())

        train_dict[epoch] = {k: np.mean(v) for k, v in running_loss.items()}
        
        mse_scaled = mean_squared_error(
            target[:, :, 0].cpu().detach().numpy(), predictions.cpu().detach().numpy()
        )
        mae_scaled = mean_absolute_error(
            target[:, :, 0].cpu().detach().numpy(), predictions.cpu().detach().numpy()
        )
        rmse_scaled = np.sqrt(mse_scaled)
        r2_scaled = r2_score(
            target[:, :, 0].cpu().detach().numpy(), predictions.cpu().detach().numpy()
        )
        my_r2_scaled = criterion.r2(
            target[:, :, 0].cpu().detach(), predictions.cpu().detach()
        )

        # inverse transform
        target = scaler.inverse_transform(
            target[:, :, 0].cpu().detach().numpy().reshape(-1, 1)
        )
        predictions = scaler.inverse_transform(
            predictions.cpu().detach().numpy().reshape(-1, 1)
        )
        mse = mean_squared_error(target, predictions)
        mae = mean_absolute_error(target, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(target, predictions)
        my_r2 = criterion.r2(target, predictions)

        measures_train.setdefault("mse", []).append(mse)
        measures_train.setdefault("mae", []).append(mae)
        measures_train.setdefault("rmse", []).append(rmse)
        measures_train.setdefault("r2", []).append(r2)
        measures_train.setdefault("my_r2", []).append(my_r2)
        measures_train.setdefault("mse_scaled", []).append(mse_scaled)
        measures_train.setdefault("mae_scaled", []).append(mae_scaled)
        measures_train.setdefault("rmse_scaled", []).append(rmse_scaled)
        measures_train.setdefault("r2_scaled", []).append(r2_scaled)
        measures_train.setdefault("my_r2_scaled", []).append(my_r2_scaled)

        # validation
        model.eval()
        running_loss = {}
        for x_ts, y, idx in test_loader:
            x_ts = torch.tensor(x_ts).to(device)
            target = torch.tensor(y).to(device).squeeze()
            idx = torch.tensor(idx).to(device)

            with torch.no_grad():
                predictions = model(x_ts, idx, **graph_inputs)
                loss_dict = criterion(predictions, target[:, :, 0])
                for k, v in loss_dict.items():
                    running_loss.setdefault(k, []).append(v.item())

        val_dict[epoch] = {k: np.mean(v) for k, v in running_loss.items()}

        mse_scaled = mean_squared_error(
            target[:, :, 0].cpu().detach().numpy(), predictions.cpu().detach().numpy()
        )
        mae_scaled = mean_absolute_error(
            target[:, :, 0].cpu().detach().numpy(), predictions.cpu().detach().numpy()
        )
        rmse_scaled = np.sqrt(mse_scaled)
        r2_scaled = r2_score(
            target[:, :, 0].cpu().detach().numpy(), predictions.cpu().detach().numpy()
        )
        my_r2_scaled = criterion.r2(
            target[:, :, 0].cpu().detach(), predictions.cpu().detach()
        )

        # inverse transform
        target = scaler.inverse_transform(
            target[:, :, 0].cpu().detach().numpy().reshape(-1, 1)
        )
        predictions = scaler.inverse_transform(
            predictions.cpu().detach().numpy().reshape(-1, 1)
        )
        mse = mean_squared_error(target, predictions)
        mae = mean_absolute_error(target, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(target, predictions)
        my_r2 = criterion.r2(target, predictions)

        measures_val.setdefault("mse", []).append(mse)
        measures_val.setdefault("mae", []).append(mae)
        measures_val.setdefault("rmse", []).append(rmse)
        measures_val.setdefault("r2", []).append(r2)
        measures_val.setdefault("my_r2", []).append(my_r2)
        measures_val.setdefault("mse_scaled", []).append(mse_scaled)
        measures_val.setdefault("mae_scaled", []).append(mae_scaled)
        measures_val.setdefault("rmse_scaled", []).append(rmse_scaled)
        measures_val.setdefault("r2_scaled", []).append(r2_scaled)
        measures_val.setdefault("my_r2_scaled", []).append(my_r2_scaled)


        if val_dict[epoch]["loss"] < val_best:
            val_best = val_dict[epoch]["loss"]
            wait = 0
            best_epoch = epoch
            best_state = model.state_dict()
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                model.load_state_dict(best_state)
                break

    model.load_state_dict(best_state)

    # plot losses
    train_losses = [v["loss"] for k, v in train_dict.items()]
    val_losses = [v["loss"] for k, v in val_dict.items()]
    min_loss = np.min([val_best, np.min(train_losses)])
    max_mean = np.max(
        [
            np.mean(train_losses) + 1 * np.std(train_losses),
            np.mean(val_losses) + 1 * np.std(val_losses),
        ]
    )
    exponent = int(np.log10(min_loss)) - 2
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.ylim(-1 * 10**exponent + min_loss, max_mean)
    plt.legend()
    plt.show()

    best_results_train = {
        **train_dict[best_epoch],
        "mse": measures_train["mse"][best_epoch],
        "mae": measures_train["mae"][best_epoch],
        "rmse": measures_train["rmse"][best_epoch],
        "r2": measures_train["r2"][best_epoch],
        "my_r2": measures_train["my_r2"][best_epoch],
        "mse_scaled": measures_train["mse_scaled"][best_epoch],
        "mae_scaled": measures_train["mae_scaled"][best_epoch],
        "rmse_scaled": measures_train["rmse_scaled"][best_epoch],
        "r2_scaled": measures_train["r2_scaled"][best_epoch],
        "my_r2_scaled": measures_train["my_r2_scaled"][best_epoch],
    }

    best_results_val = {
        **val_dict[best_epoch],
        "mse": measures_val["mse"][best_epoch],
        "mae": measures_val["mae"][best_epoch],
        "rmse": measures_val["rmse"][best_epoch],
        "r2": measures_val["r2"][best_epoch],
        "my_r2": measures_val["my_r2"][best_epoch],
        "mse_scaled": measures_val["mse_scaled"][best_epoch],
        "mae_scaled": measures_val["mae_scaled"][best_epoch],
        "rmse_scaled": measures_val["rmse_scaled"][best_epoch],
        "r2_scaled": measures_val["r2_scaled"][best_epoch],
        "my_r2_scaled": measures_val["my_r2_scaled"][best_epoch],
    }

    return (model, train_dict, val_dict, measures_train, measures_val, best_results_train, best_results_val)


if __name__ == '__main__':

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    # load data
    rifiuti_puglia = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data", "rifiuti_puglia.csv"))

    # load x_features
    features_comuni_global = np.load(os.path.join(os.path.dirname(__file__), "../data", "features_comuni_global.npy"), allow_pickle=True).item()
    
    # load graph
    with open(os.path.join(os.path.dirname(__file__), "../data", "newgraph.pkl"), "rb") as f:
        graph = pickle.load(f)

    laplacian = np.load(os.path.join(os.path.dirname(__file__), "../data", "normalized_laplacian_version.npy"), allow_pickle=True)

    # load idx_cluster
    idx_cluster = np.load(os.path.join(os.path.dirname(__file__), "../data", "idx_cluster.npy"), allow_pickle=True).item()

    # load scalers
    scaler_dict = np.load(os.path.join(os.path.dirname(__file__), "../data", "scaler_dict.npy"), allow_pickle=True).item()
    
    # load train_loader
    train_loader = np.load(os.path.join(os.path.dirname(__file__), "../data", "train_loader.npy"), allow_pickle=True).item()

    # load test_loader
    test_loader = np.load(os.path.join(os.path.dirname(__file__), "../data", "test_loader.npy"), allow_pickle=True).item()

    # get graph inputs
    label_to_index = {
        label: idx for idx, label in enumerate(rifiuti_puglia["comune"].unique())
    }
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    
    edge_index = np.array(list(graph.edges()))
    edge_idx = np.array(
        [[label_to_index[src], label_to_index[dst]] for src, dst in edge_index]
    ).T

    edge_attr = nx.get_edge_attributes(graph, "weight")
    edge_attr = [edge_attr[(src, dst)] for src, dst in edge_index]
    edge_attr = np.array(edge_attr)
    edge_attr = edge_attr.reshape(-1, 1)

    edge_idx = torch.from_numpy(edge_idx).long()
    edge_attr = torch.from_numpy(edge_attr).float()
    laplacian = torch.from_numpy(laplacian).float()

    # nodes dictionary
    indexes = [i for i in range(0, len(rifiuti_puglia["comune"].unique()))]
    zip_iterator = zip(rifiuti_puglia["comune"].unique(), indexes)
    nodes_dict = dict(zip_iterator)

    x_features = torch.zeros(210, 4).float()
    for comune in rifiuti_puglia["comune"].unique():
        x_features[label_to_index[comune]] = torch.from_numpy(
            np.array(
                [
                    features_comuni_global[comune]["popolazione"],
                    features_comuni_global[comune]["reddito_procapite"],
                    features_comuni_global[comune]["latitudine_scaled"],
                    features_comuni_global[comune]["longitudine_scaled"],
                ]
            )
        ).float()

    graph_inputs = {
    "x_features": x_features,
    "x_edge_idx": edge_idx,
    "x_edge_attr": edge_attr,
    "x_adj": laplacian,
    }
    graph_inputs = {kgi: gi.to(device) for kgi, gi in graph_inputs.items()}

    # train models
    window = 24
    step_window = 1
    horizon = 18

    models = {
    "Cluster 0": {
        "model": model_complete0(horizon),
        "lr": 0.001,
        "patience": 100,
        "epochs": 1500,
        "name": "cluster 0",
        "wd": 1e-5,
    },
    "Cluster 1": {
        "model": model_complete1(horizon),
        "lr": 0.001,
        "patience": 100,
        "epochs": 1500,
        "name": "cluster 1",
        "wd": 1e-4,
    },
    "Cluster 2": {
        "model": model_complete2(horizon),
        "lr": 0.001,
        "patience": 100,
        "epochs": 1500,
        "name": "cluster 2",
        "wd": 1e-4,
    },
    "Cluster 3": {
        "model": model_complete3(horizon),
        "lr": 0.001,
        "patience": 100,
        "epochs": 1500,
        "name": "cluster 3",
        "wd": 1e-3,
    }
    }    

    train_losses = {}
    val_losses = {}
    metric_train = {}
    metric_val = {}
    best_metric_train = {}
    best_metric_val = {}

    for idx_clust in idx_cluster.keys():
        model_dict = models[idx_clust]
        (
            models[idx_clust],
            train_losses[idx_clust],
            val_losses[idx_clust],
            metric_train[idx_clust],
            metric_val[idx_clust],
            best_metric_train[idx_clust],
            best_metric_val[idx_clust],
        ) = train(
            train_loader[idx_clust],
            test_loader[idx_clust],
            scaler_dict[idx_clust]["scaler"],
            model_dict,
            graph_inputs,
        )

    display(pd.DataFrame(best_metric_train).T)
    display(pd.DataFrame(best_metric_val).T)

    # test models
    df_test = np.load("data/df_test.npy", allow_pickle=True).item()

    compared_model = "GCN"
    risultati = {}
    preds_gcn = {}
    criterion = my_loss(reduction="mean")

    for i, idx_clust in enumerate(tqdm(idx_cluster.keys())):

        folder_path = f"output/h={horizon}"
        os.makedirs(folder_path, exist_ok=True)

        model = models[idx_clust]
        model = model.to(device)
        model.eval()
        scaler = scaler_dict[idx_clust]["scaler"]

        for comune in idx_cluster[idx_clust]:
            X_test_comune = []
            y_test_comune = []
            assert comune in rifiuti_puglia["comune"].unique(), "Comune not found"

            col = ["tot_rsu", "trend", "seasonal", "residual"]

            comune_data = df_test[idx_clust][df_test[idx_clust]["comune"] == comune]
            X_test_comune, y_test_comune, id_comune_comune = create_windows(
                comune_data, window, horizon, nodes_dict, col, step_window
            )
            test_comune = TensorDataset(
                torch.tensor(X_test_comune, dtype=torch.float32),
                torch.tensor(y_test_comune, dtype=torch.float32),
                torch.tensor(id_comune_comune, dtype=torch.long),
            )
            test_loader_comune = DataLoader(test_comune, shuffle=False, batch_size=256)

            # predict
            preds_comune = []
            true_values_comune = []
            for x, y, idx in test_loader_comune:
                x = x.to(device)
                y = y.to(device)
                idx = idx.to(device)
                with torch.no_grad():
                    y_pred = model(x, idx, **graph_inputs)

                preds_comune.append(y_pred.cpu().numpy())
                true_values_comune.append(y[:, :, 0].cpu().numpy())

            preds_comune = np.concatenate(preds_comune).squeeze()
            true_values_comune = np.concatenate(true_values_comune).squeeze()

            preds_gcn.setdefault(idx_clust, {})[comune] = {}
            preds_gcn[idx_clust][comune]["preds_scaled"] = preds_comune
            preds_gcn[idx_clust][comune]["true_scaled"] = true_values_comune

            # metrics
            mse_comune_scaled = mean_squared_error(true_values_comune, preds_comune)
            mae_comune_scaled = mean_absolute_error(true_values_comune, preds_comune)
            rmse_comune_scaled = np.sqrt(mse_comune_scaled)
            r2_comune_scaled = r2_score(
                true_values_comune.squeeze(), preds_comune.squeeze()
            )
            my_r2_comune_scaled = criterion.r2(
                true_values_comune.squeeze(), preds_comune.squeeze()
            ).numpy()

            risultati.setdefault(idx_clust, {})[comune] = {}
            risultati[idx_clust][comune]["mse_scaled"] = mse_comune_scaled
            risultati[idx_clust][comune]["mae_scaled"] = mae_comune_scaled
            risultati[idx_clust][comune]["rmse_scaled"] = rmse_comune_scaled
            risultati[idx_clust][comune]["r2_scaled"] = r2_comune_scaled
            risultati[idx_clust][comune]["my_r2_scaled"] = my_r2_comune_scaled

            # inverse transform
            preds_comune_unscaled = scaler.inverse_transform(
                preds_comune.reshape(-1, 1)
            ).reshape(-1, horizon)
            true_values_comune_unscaled = scaler.inverse_transform(
                true_values_comune.reshape(-1, 1)
            ).reshape(-1, horizon)

            preds_gcn[idx_clust][comune]["preds"] = preds_comune_unscaled
            preds_gcn[idx_clust][comune]["true"] = true_values_comune_unscaled

            mse_comune = mean_squared_error(
                true_values_comune_unscaled, preds_comune_unscaled
            )
            mae_comune = mean_absolute_error(
                true_values_comune_unscaled, preds_comune_unscaled
            )
            rmse_comune = np.sqrt(mse_comune)
            r2_comune = r2_score(
                true_values_comune_unscaled.squeeze(), preds_comune_unscaled.squeeze()
            )
            my_r2_comune = criterion.r2(
                true_values_comune_unscaled.squeeze(), preds_comune_unscaled.squeeze()
            ).numpy()

            risultati[idx_clust][comune]["mse"] = mse_comune
            risultati[idx_clust][comune]["mae"] = mae_comune
            risultati[idx_clust][comune]["rmse"] = rmse_comune
            risultati[idx_clust][comune]["r2"] = r2_comune
            risultati[idx_clust][comune]["my_r2"] = my_r2_comune


        
