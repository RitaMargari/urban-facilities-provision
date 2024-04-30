import torch 
import torch.nn as nn
import numpy as np

from modules.metrics import CPC
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model._glm.glm import PoissonRegressor

def train_func(linear, train_data):

    data = next(iter(train_data))

    y_train = data.y.numpy()
    x_s = data.x_s[data.edge_index[0], 1].unsqueeze(1)
    x_t = data.x_t[data.edge_index[1], 1].unsqueeze(1)
    dist = data.edge_attr.unsqueeze(1) + 1
    X_train = torch.concat([x_s, x_t, dist], 1).numpy() + 10e-5

    if type(linear) == PoissonRegressor:
        mask = (X_train[:, 0] != 0) & (X_train[:, 1] != 0)
    else:
        mask = (X_train[:, 0] != 0) & (X_train[:, 1] != 0) & (y_train != 0)

    X_train = X_train[mask]
    y_train = y_train[mask]

    X_train = np.log(X_train)
    if not type(linear) == PoissonRegressor: y_train = np.log(y_train) 

    linear.fit(X_train, y_train)

    return linear

def val_func(linear, test_data, return_y=False):

    data = next(iter(test_data))

    y_test = data.y.numpy()
    x_s = data.x_s[data.edge_index[0], 1].unsqueeze(1)
    x_t = data.x_t[data.edge_index[1], 1].unsqueeze(1)
    dist = data.edge_attr.unsqueeze(1) + 1
    X_test = torch.concat([x_s, x_t, dist], 1).numpy() + 10e-5

    mask = (X_test[:, 0] != 0) & (X_test[:, 1] != 0)
    X_test = X_test[mask]
    y_test = y_test[mask]
    X_test = np.log(X_test)

    y_pred = linear.predict(X_test)
    if not type(linear) == PoissonRegressor: y_pred = np.exp(y_pred).astype(int)
    
    if return_y:
        return y_pred
    else:
        poisson_loss = nn.PoissonNLLLoss(log_input=False)
        return {
            "valid_loss": poisson_loss(torch.tensor(y_pred), torch.tensor(y_test)),
            # "valid_loss": mean_squared_error(y_test, y_pred),
            "valid_r2": r2_score(y_test, y_pred),
            "valid_cpc": CPC(y_test, y_pred)
            }