import torch_geometric.nn as pyg_nn
import numpy as np
import math

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch

from modules.metrics import r2_loss, weighted_mse_loss

class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, heads=3, aggr=None):
        super(GNNStack, self).__init__()

        self.num_layers = 2
        self.dropout = dropout

        self.norm = nn.ModuleList([nn.LayerNorm(hidden_dim * heads),  nn.LayerNorm(hidden_dim)])
        self.kwargs = {"heads": heads, "flow": "source_to_target", "edge_dim": 1, "add_self_loops": False}
        self.convs = nn.ModuleList([
            pyg_nn.GATConv(input_dim, hidden_dim, **self.kwargs),
            pyg_nn.GATConv((hidden_dim * heads, hidden_dim * heads), hidden_dim, **self.kwargs, concat=False)
            ])

    def forward(self, data):
        x_s, x_t, edge_index, edge_weight = data.x_s, data.x_t, data.edge_index, data.edge_attr.unsqueeze(-1)
        edge_index_reverse = torch.concat((edge_index[1].unsqueeze(1), edge_index[0].unsqueeze(1)), 1).T
    
        for i in range(self.num_layers):

            kwargs_st = {"size": (x_s.size(0), x_t.size(0)), "return_attention_weights": True}
            kwargs_ts = {"size": (x_t.size(0), x_s.size(0)), "return_attention_weights": True}
            x_new_t, at_t = self.convs[i]((x_s, x_t), edge_index, edge_weight, **kwargs_st)
            x_new_s, at_s = self.convs[i]((x_t, x_s), edge_index_reverse, edge_weight, **kwargs_ts)

            x_t = F.leaky_relu(x_new_t)
            x_t = F.dropout(x_t, p=self.dropout, training=self.training)
            x_t = self.norm[i](x_t)

            x_s = F.leaky_relu(x_new_s)
            x_s = F.dropout(x_s, p=self.dropout, training=self.training)
            x_s = self.norm[i](x_s)

        return x_s, at_s, x_t, at_t


class FNNStack_v0(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(FNNStack_v0, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(self.num_layers - 2):
            self.lins.append(nn.Linear(hidden_dim, hidden_dim))
        self.lins.append(nn.Linear(hidden_dim, output_dim))

        self.norm = nn.ModuleList()
        for l in range(self.num_layers):
            self.norm.append(nn.LayerNorm(hidden_dim))

    def forward(self, emb_s, emb_t, at_s, at_t, data):

        x_s, x_t, edge_index, edge_weight = data.x_s, data.x_t, data.edge_index, data.edge_attr.unsqueeze(-1)

        x_s_d = x_s[:, 1][edge_index[0]].unsqueeze(1)
        x_s_c = x_t[:, 1][edge_index[1]].unsqueeze(1)
        emb_s = emb_s[edge_index[0]]
        emb_t = emb_t[edge_index[1]]
        atten_s = at_s[1].mean(1).unsqueeze(-1)
        atten_t = at_t[1].mean(1).unsqueeze(-1)

        y = torch.cat((emb_s, emb_t, x_s_d, x_s_c, atten_s, atten_t, edge_weight), axis=1)
        for i in range(self.num_layers - 1):
            y = self.lins[i](y) 
            y = nn.functional.leaky_relu(y)
            y = F.dropout(y, p=self.dropout, training=self.training)
            y = self.norm[i](y)

        y = self.lins[-1](y)
        y = torch.relu(y).squeeze()
        return y


class FNNStack_v1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(FNNStack_v1, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(self.num_layers - 2):
            self.lins.append(nn.Linear(hidden_dim, hidden_dim))
        self.lins.append(nn.Linear(hidden_dim, output_dim))

        self.norm = nn.ModuleList()
        for l in range(self.num_layers):
            self.norm.append(nn.LayerNorm(hidden_dim))

    def normalize(self, x_s, x_t, edge_index, y):

        y_sum_i = torch.zeros_like(x_s[:, 1], dtype=y.dtype).index_add_(0, edge_index[0], y)
        y_norm_i = y * x_s[:, 1][edge_index[0, :]] / (y_sum_i[edge_index[0]] + 3.4028e-38)
        y_sum_j = torch.zeros_like(x_t[:, 1], dtype=y.dtype).index_add_(0, edge_index[1], y)
        y_norm_j = y * x_t[:, 1][edge_index[1, :]] / (y_sum_j[edge_index[1]] + 3.4028e-38)

        y_new = torch.min(y_norm_i, y_norm_j)
        return y_new

    def forward(self, emb_s, emb_t, at_s, at_t, data):

        x_s, x_t, edge_index, edge_weight = data.x_s, data.x_t, data.edge_index, data.edge_attr.unsqueeze(-1)

        x_s_d = x_s[:, 1][edge_index[0]].unsqueeze(1)
        x_s_c = x_t[:, 1][edge_index[1]].unsqueeze(1)
        emb_s = emb_s[edge_index[0]]
        emb_t = emb_t[edge_index[1]]
        atten_s = at_s[1].mean(1).unsqueeze(-1)
        atten_t = at_t[1].mean(1).unsqueeze(-1)

        y = torch.cat((emb_s, emb_t, x_s_d, x_s_c, atten_s, atten_t, edge_weight), axis=1)
        for i in range(self.num_layers - 1):
            y = self.lins[i](y) 
            y = nn.functional.leaky_relu(y)
            y = F.dropout(y, p=self.dropout, training=self.training)
            y = self.norm[i](y)

        y = self.lins[-1](y)
        y = torch.relu(y).squeeze()
        y_norm = self.normalize(x_s, x_t, edge_index, y)
        return y_norm


class FNNStack_v2(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim, num_layers, num_layers_norm, dropout):
        super(FNNStack_v2, self).__init__()

        self.num_layers = num_layers
        self.num_layers_norm = num_layers_norm
        self.dropout = dropout

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(input_dim, hidden_dim_1))
        for _ in range(self.num_layers - 2):
            self.lins.append(nn.Linear(hidden_dim_1, hidden_dim_1))
        self.lins.append(nn.Linear(hidden_dim_1, output_dim))

        self.norm = nn.ModuleList()
        for l in range(self.num_layers):
            self.norm.append(nn.LayerNorm(hidden_dim_1))

        self.factors_model = nn.Sequential(
          nn.Linear(5, hidden_dim_2),
          nn.ReLU(),
          nn.Dropout(p=self.dropout),
          nn.LayerNorm(hidden_dim_2),
          nn.Linear(hidden_dim_2, 1),
          nn.ReLU()
        )

    def forward(self, emb_s, emb_t, at_s, at_t, data):

        x_s, x_t, edge_index, edge_weight = data.x_s, data.x_t, data.edge_index, data.edge_attr.unsqueeze(-1)

        x_s_d = x_s[:, 1][edge_index[0]].unsqueeze(1)
        x_s_c = x_t[:, 1][edge_index[1]].unsqueeze(1)
        emb_s = emb_s[edge_index[0]]
        emb_t = emb_t[edge_index[1]]
        atten_s = at_s[1].mean(1).unsqueeze(-1)
        atten_t = at_t[1].mean(1).unsqueeze(-1)

        y = torch.cat((emb_s, emb_t, x_s_d, x_s_c, atten_s, atten_t, edge_weight), axis=1)
        y = F.normalize(y)
        for i in range(self.num_layers - 1):
            y = self.lins[i](y) 
            y = nn.functional.leaky_relu(y)
            y = F.dropout(y, p=self.dropout, training=self.training)
            y = self.norm[i](y)

        y = self.lins[-1](y)
        y = torch.relu(y).squeeze()

        for i in range(self.num_layers_norm):

            y_sum_i = torch.zeros_like(x_s[:, 1], dtype=y.dtype).index_add_(0, edge_index[0], y)[edge_index[0]].unsqueeze(1)
            y_sum_j = torch.zeros_like(x_t[:, 1], dtype=y.dtype).index_add_(0, edge_index[1], y)[edge_index[1]].unsqueeze(1)
            coef = torch.cat((y.unsqueeze(1), y_sum_i, x_s_d, y_sum_j, x_s_c), axis=1)
            coef = F.normalize(coef)
            coef = self.factors_model(coef).squeeze()
            y = y * coef 

        return y * coef

def train_func(gnn_model, fnn_model, train_loader, valid_loader, epochs, writer=None, output=False):

    params = list(fnn_model.parameters()) + list(gnn_model.parameters())
    optimize = optim.Adam(params,  lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimize, factor=0.9, min_lr=0.001)

    # train
    for epoch in range(epochs + 1):
        train_loss = []
        train_r2 = []
        for train_data in train_loader:
            optimize.zero_grad()
            fnn_model.train()
            gnn_model.train()

            emb_s, at_s, emb_t, at_t = gnn_model(train_data)
            predict_y = fnn_model(emb_s, emb_t, at_s, at_t, train_data)

            loss = F.mse_loss(predict_y, train_data.y)
            r2 = r2_loss(predict_y, train_data.y)
            train_loss.append(loss)
            train_r2.append(r2)

            loss.backward()
            optimize.step()

        t_metrics = {"train_loss": sum(train_loss)/len(train_loss), "train_r2": sum(train_r2)/len(train_r2)}
        v_metrics = val_func(valid_loader, gnn_model, fnn_model)
        scheduler.step(v_metrics["valid_loss"])      

        if epoch % 10 == 0:

            if output: print(
                "Epoch {}. TRAIN: loss {:.4f}, r2: {:.4f}. ".format(epoch, t_metrics["train_loss"], t_metrics["train_r2"]) + \
                "VALIDATION loss: {:.4f}, r2: {:.4f}. ".format(v_metrics["valid_loss"],  v_metrics["valid_r2"]) + \
                "Lr: {:.5f}".format(optimize.param_groups[0]["lr"]), 
            )
            if writer: 
                for name, v_metric in v_metrics.items(): writer.add_scalar(name, v_metric, epoch)
                for name, v_metric in t_metrics.items(): writer.add_scalar(name, v_metric, epoch)
            # save_ckp(epoch, fnn_model, optimize, datetime_now + f"_epoch_{epoch}", f_path=path)

    return [gnn_model, fnn_model]


def val_func(valid_loader, gnn_model, fnn_model, return_y=False):

    valid_loss = []
    valid_r2 = []
    valid_data = next(iter(valid_loader))
    with torch.no_grad():
        fnn_model.eval()
        gnn_model.eval()

        emb_s, at_s, emb_t, at_t = gnn_model(valid_data)
        predict_y = fnn_model(emb_s, emb_t, at_s, at_t, valid_data)
        
        valid_loss.append(F.mse_loss(predict_y, valid_data.y))
        valid_r2.append(r2_loss(predict_y, valid_data.y))

    if return_y:
        return predict_y
    else:
        return {"valid_loss": np.mean(valid_loss), "valid_r2": np.mean(valid_r2)}