import torch.nn.functional as F
import torch.nn as nn
import torch

# torch.manual_seed(0)

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

    def forward(self, data):

        x_s, x_t, edge_index, edge_weight = data.x_s, data.x_t, data.edge_index, data.edge_attr
        edge_weight = edge_weight.unsqueeze(-1)

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

    def forward(self, data):

        x_s, x_t, edge_index, edge_weight = data.x_s, data.x_t, data.edge_index, data.edge_attr
        edge_weight = edge_weight.unsqueeze(-1)

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