import torch 
import numpy as np

from sklearn.metrics import r2_score, mean_squared_error

def train_func(linear, train_data):

    data = next(iter(train_data))
    y_train = data.y

    x_s = data.x_s[data.edge_index[0], 1].unsqueeze(1)
    x_t = data.x_t[data.edge_index[1], 1].unsqueeze(1)
    dist = data.edge_attr.unsqueeze(1)
    X_train = torch.concat([x_s, x_t, dist], 1)

    linear.fit(X_train, y_train)

    return linear

def val_func(linear, test_data, return_y=False):

    data = next(iter(test_data))
    y_test = data.y

    x_s = data.x_s[data.edge_index[0], 1].unsqueeze(1)
    x_t = data.x_t[data.edge_index[1], 1].unsqueeze(1)
    dist = data.edge_attr.unsqueeze(1)
    X_test = torch.concat([x_s, x_t, dist], 1)

    y_pred = linear.predict(X_test)
    y_pred = np.where(y_pred > 0, y_pred, 0)

    if return_y:
        return y_pred
    else:
        return {"valid_loss": mean_squared_error(y_test, y_pred), "valid_r2": r2_score(y_test, y_pred)} 