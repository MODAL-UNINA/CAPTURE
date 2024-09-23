import torch
import numpy as np
import torch.nn as nn
import torch_geometric.nn as gnn
from torch.autograd import Variable

device_number = 0
device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")

# RNN
class RNN_model(nn.Module):
    def __init__(self, output_size, final_attivation=None, num_layers=2):
        super(RNN_model, self).__init__()
        self.output_size = output_size
        self.num_layers = num_layers
        self.final_attivation = final_attivation
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=64,
            num_layers=self.num_layers,
            dropout=0,
            bidirectional=False,
        )

        self.fc = nn.LazyLinear(output_size)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, idx):
        x = x.unsqueeze(-1)
        x = x.permute(1, 0, 2)
        h_0 = Variable(torch.zeros(self.num_layers, x.shape[1], 64).to(device))
        out, _ = self.rnn(x, h_0)
        out = out.permute(1, 0, 2)

        out = self.fc(out.reshape(out.shape[0], -1))

        if self.final_attivation == "tanh":
            out = self.tanh(out)

        if self.final_attivation == "sigmoid":
            out = self.sigmoid(out)

        if self.final_attivation == "relu":
            out = self.relu(out)

        return out


# LSTM
class LSTM_model(nn.Module):
    def __init__(self, output_size, final_attivation=None, num_layers=2):
        super(LSTM_model, self).__init__()
        self.output_size = output_size
        self.num_layers = num_layers
        self.final_attivation = final_attivation
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=self.num_layers,
            dropout=0,
            bidirectional=False,
        )

        self.fc = nn.LazyLinear(output_size)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, idx):
        x = x.unsqueeze(-1)
        x = x.permute(1, 0, 2)
        h_0 = Variable(torch.zeros(self.num_layers, x.shape[1], 64).to(device))
        c_0 = Variable(torch.zeros(self.num_layers, x.shape[1], 64).to(device))
        out, _ = self.lstm(x, (h_0, c_0))
        out = out.permute(1, 0, 2)

        out = self.fc(out.reshape(out.shape[0], -1))

        if self.final_attivation == "tanh":
            out = self.tanh(out)

        if self.final_attivation == "sigmoid":
            out = self.sigmoid(out)

        if self.final_attivation == "relu":
            out = self.relu(out)

        return out


# GCN
class GCN_model(nn.Module):
    def __init__(self, output_size, is_cuda=True):
        super(GCN_model, self).__init__()
        self.output_size = output_size
        self.is_cuda = is_cuda
        self.gcn = gnn.GCNConv(-1, 210)
        self.fc = nn.LazyLinear(output_size)
        self.tanh = nn.Tanh()

    def forward(self, x_features, x_edge_idx, x_edge_attr, x_adj=None):
        x = self.gcn(x_features, x_edge_idx, edge_weight=x_edge_attr)
        out = self.fc(x)
        out = self.tanh(out)

        return out


# MLP
class MLP_model(nn.Module):
    def __init__(self, horizon):
        self.horizon = horizon
        super(MLP_model, self).__init__()
        self.fc1 = nn.LazyLinear(32)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, horizon, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)

        return x


# MODEL FIRST CLUSTER
class model_complete0(nn.Module):
    def __init__(self, output_size, final_attivation=None):
        super(model_complete0, self).__init__()
        self.output_size = output_size
        self.final_attivation = final_attivation
        self.lstm1 = LSTM_model(output_size, final_attivation="tanh", num_layers=2)
        self.lstm2 = LSTM_model(output_size, final_attivation="tanh", num_layers=2)
        self.rnn = RNN_model(output_size, final_attivation="relu", num_layers=2)
        self.mlp = MLP_model(output_size)
        self.gcn = GCN_model(output_size, is_cuda=True)

    def forward(self, x, idx, x_features, x_edge_idx, x_edge_attr, x_adj):
        out1 = self.lstm1(x[:, :, 1], idx)
        out2 = self.lstm2(x[:, :, 2], idx)
        out3 = self.rnn(x[:, :, 3], idx)
        out = torch.cat([out1, out2, out3], dim=1)
        x_gcn = self.gcn(x_features, x_edge_idx, x_edge_attr, x_adj)
        x_gcn = x_gcn[idx]

        out = torch.cat([out, x_gcn], dim=1)
        out = self.mlp(out)

        return out


# MODEL SECOND CLUSTER
class model_complete1(nn.Module):
    def __init__(self, output_size, final_attivation=None):
        super(model_complete1, self).__init__()
        self.output_size = output_size
        self.final_attivation = final_attivation
        self.lstm1 = LSTM_model(output_size, final_attivation="tanh", num_layers=2)
        self.lstm2 = LSTM_model(output_size, final_attivation="tanh", num_layers=2)
        self.rnn = RNN_model(output_size, final_attivation="relu", num_layers=2)
        self.mlp = MLP_model(output_size)
        self.gcn = GCN_model(output_size, is_cuda=True)

    def forward(self, x, idx, x_features, x_edge_idx, x_edge_attr, x_adj):
        out1 = self.lstm1(x[:, :, 1], idx)
        out2 = self.lstm2(x[:, :, 2], idx)
        out3 = self.rnn(x[:, :, 3], idx)
        out = torch.cat([out1, out2, out3], dim=1)
        x_gcn = self.gcn(x_features, x_edge_idx, x_edge_attr, x_adj)
        x_gcn = x_gcn[idx]

        out = torch.cat([out, x_gcn], dim=1)
        out = self.mlp(out)

        return out


# MODEL THIRD CLUSTER
class model_complete2(nn.Module):
    def __init__(self, output_size, final_attivation=None):
        super(model_complete2, self).__init__()
        self.output_size = output_size
        self.final_attivation = final_attivation
        self.lstm1 = LSTM_model(output_size, final_attivation="tanh", num_layers=2)
        self.lstm2 = LSTM_model(output_size, final_attivation="tanh", num_layers=2)
        self.rnn = RNN_model(output_size, final_attivation="relu", num_layers=2)
        self.mlp = MLP_model(output_size)
        self.gcn = GCN_model(output_size, is_cuda=True)

    def forward(self, x, idx, x_features, x_edge_idx, x_edge_attr, x_adj):
        out1 = self.lstm1(x[:, :, 1], idx)
        out2 = self.lstm2(x[:, :, 2], idx)
        out3 = self.rnn(x[:, :, 3], idx)
        out = torch.cat([out1, out2, out3], dim=1)
        x_gcn = self.gcn(x_features, x_edge_idx, x_edge_attr, x_adj)
        x_gcn = x_gcn[idx]

        out = torch.cat([out, x_gcn], dim=1)
        out = self.mlp(out)

        return out


# MODEL FOURTH CLUSTER
class model_complete3(nn.Module):
    def __init__(self, output_size, final_attivation=None):
        super(model_complete3, self).__init__()
        self.output_size = output_size
        self.final_attivation = final_attivation
        self.lstm1 = RNN_model(output_size, final_attivation="tanh", num_layers=2)
        self.lstm2 = RNN_model(output_size, final_attivation="tanh", num_layers=2)
        self.rnn = RNN_model(output_size, final_attivation="relu", num_layers=2)
        self.mlp = MLP_model(output_size)
        self.gcn = GCN_model(output_size, is_cuda=True)

    def forward(self, x, idx, x_features, x_edge_idx, x_edge_attr, x_adj):
        out1 = self.lstm1(x[:, :, 1], idx)
        out2 = self.lstm2(x[:, :, 2], idx)
        out3 = self.rnn(x[:, :, 3], idx)
        out = torch.cat([out1, out2, out3], dim=1)
        x_gcn = self.gcn(x_features, x_edge_idx, x_edge_attr, x_adj)
        x_gcn = x_gcn[idx]

        out = torch.cat([out, x_gcn], dim=1)
        out = self.mlp(out)

        return out


#LOSS
class my_loss(nn.Module):
    def __init__(self, reduction="mean") -> None:
        super(my_loss, self).__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    @staticmethod
    def r2_loss(target, output):
        target_mean = torch.mean(target)
        ss_tot = torch.sum((target - target_mean) ** 2)
        ss_res = torch.sum((target - output) ** 2)
        r2 = 1 - ss_res / ss_tot
        return 1 - r2

    @staticmethod
    def r2(target, output):
        if isinstance(target, np.ndarray):
            target = torch.tensor(target)
        if isinstance(output, np.ndarray):
            output = torch.tensor(output)

        target_mean = torch.mean(target)
        ss_tot = torch.sum((target - target_mean) ** 2)
        ss_res = torch.sum((target - output) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2

    @staticmethod
    def annual_error(target, output):
        annual_error = torch.mean(torch.abs(target.sum(1) - output.sum(1)), dim=0)
        return annual_error

    def forward(self, target, output):
        mse = self.mse(target, output)
        r2 = self.r2_loss(target, output)
        annual_error = self.annual_error(target, output)

        loss = mse + r2 + annual_error

        return {
            "loss": loss,
            "mse_loss": mse,
            "r2_loss": r2,
            "annual_loss": annual_error,
        }