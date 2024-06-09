import math
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import copy

from modules.metrics import r2_loss, CPC, weighted_mse_loss, nb2_loss

# torch.manual_seed(0)

class FNN_v0(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(FNN_v0, self).__init__()

        self.num_layers = num_layers

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(self.num_layers - 2):
            self.lins.append(nn.Linear(hidden_dim, hidden_dim))
        self.lins.append(nn.Linear(hidden_dim, output_dim))

        self.norm = nn.ModuleList()
        for l in range(self.num_layers):
            self.norm.append(nn.LayerNorm(hidden_dim))

        self.dropout = dropout
        self.disp = torch.nn.Parameter(torch.tensor(1.))

    def forward(self, x_s, x_t, edge_index, edge_weight):

        x_s_d = x_s[:, 1][edge_index[0]].unsqueeze(1)
        x_s_c = x_t[:, 1][edge_index[1]].unsqueeze(1)
        y = torch.cat((x_s_d, x_s_c, edge_weight), axis=1)

        for i in range(self.num_layers - 1):
            y = self.lins[i](y) 
            y = nn.functional.leaky_relu(y)
            y = F.dropout(y, p=self.dropout, training=self.training)
            y = self.norm[i](y)

        y = self.lins[-1](y)
        y = torch.relu(y).squeeze()
        return y

class FNN_v1(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(FNN_v1, self).__init__()

        self.num_layers = num_layers

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(self.num_layers - 2):
            self.lins.append(nn.Linear(hidden_dim, hidden_dim))
        self.lins.append(nn.Linear(hidden_dim, output_dim))

        self.norm = nn.ModuleList()
        for l in range(self.num_layers):
            self.norm.append(nn.LayerNorm(hidden_dim))

        self.dropout = dropout

    def normalize(self, x_s, x_t, edge_index, y):

        y_sum_i = torch.zeros_like(x_s[:, 1], dtype=y.dtype).index_add_(0, edge_index[0], y)
        y_norm_i = y * x_s[:, 1][edge_index[0, :]] / (y_sum_i[edge_index[0]] + 3.4028e-38)
        y_sum_j = torch.zeros_like(x_t[:, 1], dtype=y.dtype).index_add_(0, edge_index[1], y)
        y_norm_j = y * x_t[:, 1][edge_index[1, :]] / (y_sum_j[edge_index[1]] + 3.4028e-38)

        y_new = torch.min(y_norm_i, y_norm_j)
        return y_new

    def forward(self, x_s, x_t, edge_index, edge_weight):

        x_s_d = x_s[:, 1][edge_index[0]].unsqueeze(1)
        x_s_c = x_t[:, 1][edge_index[1]].unsqueeze(1)
        y = torch.cat((x_s_d, x_s_c, edge_weight), axis=1)

        for i in range(self.num_layers - 1):
            y = self.lins[i](y) 
            y = nn.functional.leaky_relu(y)
            y = F.dropout(y, p=self.dropout, training=self.training)
            y = self.norm[i](y)

        y = self.lins[-1](y)
        y = torch.relu(y).squeeze()
        y_norm = self.normalize(x_s, x_t, edge_index, y)
        return y_norm


class FNN_v2(nn.Module):

    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim, num_layers, dropout):
        super(FNN_v2, self).__init__()
        torch.manual_seed(0)
        self.num_layers = num_layers
        self.dropout = dropout

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(input_dim, hidden_dim_1))
        for _ in range(self.num_layers - 2):
            self.lins.append(nn.Linear(hidden_dim_1, hidden_dim_1))
        self.lins.append(nn.Linear(hidden_dim_1, output_dim))

        self.norm = nn.ModuleList()
        for l in range(self.num_layers - 1):
            self.norm.append(nn.LayerNorm(hidden_dim_1))

        self.factors_model = nn.Sequential(
          nn.Linear(4, hidden_dim_2),
          nn.ReLU(),
          nn.Dropout(p=self.dropout),
          nn.LayerNorm(hidden_dim_2),
          nn.Linear(hidden_dim_2, 1),
          nn.ReLU()
        )

    def forward(self, x_s, x_t, edge_index, edge_weight):

        x_s_d = x_s[:, 1][edge_index[0]].unsqueeze(1)
        x_s_c = x_t[:, 1][edge_index[1]].unsqueeze(1)
        y = torch.cat((x_s_d, x_s_c, edge_weight), axis=1)

        for i in range(self.num_layers - 1):
            y = self.lins[i](y) 
            y = nn.functional.leaky_relu(y)
            y = F.dropout(y, p=self.dropout, training=self.training)
            y = self.norm[i](y)

        y = self.lins[-1](y)
        y = torch.relu(y).squeeze()

        y_sum_i = torch.zeros_like(x_s[:, 1], dtype=y.dtype).index_add_(0, edge_index[0], y)[edge_index[0]].unsqueeze(1)
        y_sum_j = torch.zeros_like(x_t[:, 1], dtype=y.dtype).index_add_(0, edge_index[1], y)[edge_index[1]].unsqueeze(1)
        coef = torch.cat((x_s_d, y_sum_i, x_s_c, y_sum_j), axis=1)
        coef = self.factors_model(coef).squeeze()
        y = y * coef

        return y


def train_func(fnn_model, train_loader, valid_loader, epochs, writer=None, output=False, device='cpu'):

    optimize = optim.Adam(list(fnn_model.parameters()),  lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimize, factor=0.9, min_lr=0.001)

    best_loss = float('inf')
    best_model_state = None

    # train
    for epoch in range(epochs + 1):
        train_loss = []
        train_r2 = []
        train_cpc = []

        for train_data in train_loader:

            optimize.zero_grad()
            fnn_model.train()

            y = train_data.y.to(device)
            x_s = train_data.x_s.to(device)
            x_t = train_data.x_t.to(device)
            edge_index = train_data.edge_index.to(device)
            edge_weight = train_data.edge_attr.unsqueeze(-1).to(device)

            predict_y = fnn_model(x_s, x_t, edge_index, edge_weight)

            poisson_loss = nn.PoissonNLLLoss(log_input=False)
            loss = poisson_loss(predict_y, y)
            # loss = nb2_loss(y, torch.log(predict_y + 1e-8), fnn_model.disp)
            # loss = F.mse_loss(predict_y, y)
            r2 = r2_loss(predict_y, y)
            cpc = CPC(y, predict_y)
            train_loss.append(loss)
            train_r2.append(r2)
            train_cpc.append(cpc)

            loss.backward()
            optimize.step()

        t_metrics = {
            "train_loss": sum(train_loss)/len(train_loss), 
            "train_r2": sum(train_r2)/len(train_r2),
            "train_cpc": sum(train_cpc)/len(train_cpc)
            }        

        v_metrics = val_func(valid_loader, fnn_model, device=device)
        scheduler.step(v_metrics["valid_loss"])

        if v_metrics["valid_loss"] < best_loss:
            best_loss = v_metrics["valid_loss"]
            best_model_state = copy.deepcopy(fnn_model.state_dict())

        if epoch % 10 == 0:
            if output: print(
                "Epoch {}. TRAIN: loss {:.4f}, r2: {:.4f}. CPC: {:.4f}.".format(
                    epoch, t_metrics["train_loss"], t_metrics["train_r2"], t_metrics["train_cpc"]
                    ) + \
                "VALIDATION loss: {:.4f}, r2: {:.4f}. CPC: {:.4f}.".format(
                    v_metrics["valid_loss"],  v_metrics["valid_r2"], v_metrics["valid_cpc"]
                    ) + \
                "Lr: {:.5f}".format(optimize.param_groups[0]["lr"]), 
            )
            if writer: 
                for name, v_metric in v_metrics.items(): writer.add_scalar(name, v_metric, epoch)
                for name, v_metric in t_metrics.items(): writer.add_scalar(name, v_metric, epoch)
            # save_ckp(epoch, fnn_model, optimize, datetime_now + f"_epoch_{epoch}", f_path=path)

    fnn_model.load_state_dict(best_model_state)
    return fnn_model


def val_func(valid_loader, fnn_model, return_y=False, device='cpu'):

    valid_loss = []
    valid_r2 = []
    valid_cpc = []
    valid_data = next(iter(valid_loader))

    with torch.no_grad():
        fnn_model.eval()

        y = valid_data.y.to(device)
        x_s = valid_data.x_s.to(device)
        x_t = valid_data.x_t.to(device)
        edge_index = valid_data.edge_index.to(device)
        edge_weight = valid_data.edge_attr.unsqueeze(-1).to(device)

        mask = (x_s[edge_index[0], 1] != 0) & (x_t[edge_index[1], 1] != 0)
        y = y[mask]
        edge_index = edge_index[:, mask]
        edge_weight = edge_weight[mask]

        predict_y = fnn_model(x_s, x_t, edge_index, edge_weight)

        poisson_loss = nn.PoissonNLLLoss(log_input=False)
        
        # valid_loss.append(F.mse_loss(predict_y, y))
        valid_loss.append(poisson_loss(predict_y, y))
        valid_r2.append(r2_loss(predict_y, y))
        valid_cpc.append(CPC(y, predict_y))

    if return_y:
        return predict_y
    else:
        valid_loss = [l.cpu() for l in valid_loss]
        valid_r2 = [r.cpu() for r in valid_r2]
        return {
            "valid_loss": np.mean(valid_loss), 
            "valid_r2": np.mean(valid_r2),
            "valid_cpc": np.mean(valid_cpc)
            }