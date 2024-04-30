import os
import torch
import numpy as np

from tensorboardX import SummaryWriter
from datetime import datetime
from torch_geometric.loader.dataloader import DataLoader
from tqdm.notebook import trange


def nb2_loss(actual, log_pred, disp):
    m = 1 / torch.exp(disp)
    mu = torch.exp(log_pred)
    p = 1 / (1 + mu * torch.exp(disp))
    nll = torch.lgamma(m + actual) - torch.lgamma(actual + 1) - torch.lgamma(m)
    nll += m * torch.log(p) + actual * torch.log1p(-p)
    return -nll.mean()

def CPC(y, y_hat):

    if type(y) == torch.Tensor: y = y.detach().cpu().numpy()
    if type(y_hat) == torch.Tensor: y_hat = y_hat.detach().cpu().numpy()

    nom = 2 * np.min((y, y_hat), axis=0).sum()
    denom = y.sum() + y_hat.sum()

    return nom / denom


def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).mean()


def save_ckp(epoch, model, optimizer, datetime_now, f_path):
    checkpoint = {
    'epoch': epoch + 1,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict()
}
    f_path = os.path.join(f_path, datetime_now + '_checkpoint.pt')
    torch.save(checkpoint, f_path)


def cross_validation(dataset, num_folds, model_type, model, model_param, train_func, val_func, device='cpu',
                     epochs=None, output=False, logs=False):

    total_size = len(dataset)
    fold_size = int(total_size / num_folds)

    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    metrics = {
        "in-sample loss": [], 
        "in-sample R2": [], 
        "in-sample CPC": [],
        "out-of-sample loss": [], 
        "out-of-sample R2": [],
        "out-of-sample CPC": []
        }
    y_predicted = []
    for i in trange(num_folds):

        writer = SummaryWriter("./logs/" + datetime_now + "_" + str(i)) if logs else None

        in_sample_metrics, out_sample_metrics, y_fold = validation_step(
            i, num_folds, fold_size, total_size, dataset, model_type, model, model_param, 
            train_func, val_func, device, epochs, writer, output
            )
        
        j = 1
        while out_sample_metrics["valid_r2"] < 0:
            print(f"Bad initialized weights. The training in {i} fold is repeating. Attempt {j}...")
            in_sample_metrics, out_sample_metrics, y_fold = validation_step(
                i, num_folds, fold_size, total_size, dataset, model_type, model, model_param, 
                train_func, val_func, device, epochs, writer, output
                )
            if j == 5: break
            j += 1


        metrics["in-sample loss"].append(round(float(in_sample_metrics["valid_loss"]), 4))
        metrics["in-sample R2"].append(round(float(in_sample_metrics["valid_r2"]), 4))
        metrics["in-sample CPC"].append(round(float(in_sample_metrics["valid_cpc"]), 4))
        metrics["out-of-sample loss"].append(round(float(out_sample_metrics["valid_loss"]), 4))
        metrics["out-of-sample R2"].append(round(float(out_sample_metrics["valid_r2"]), 4))
        metrics["out-of-sample CPC"].append(round(float(out_sample_metrics["valid_cpc"]), 4))

        y_predicted.append(y_fold)

    return metrics, y_predicted

    
def validation_step(i, num_folds, fold_size, total_size, dataset, model_type, model, model_param, 
                    train_func, val_func, device, epochs, writer, output):
            
        trll = 0
        trlr = i * fold_size
        vall = trlr
        valr = i * fold_size + fold_size if i != (num_folds - 1) else total_size
        trrl = valr
        trrr = total_size

        train_left_indices = list(range(trll, trlr))
        train_right_indices = list(range(trrl, trrr))
        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(vall,valr))

        train_set = torch.utils.data.dataset.Subset(dataset, train_indices)
        val_set = torch.utils.data.dataset.Subset(dataset, val_indices)
        train_loader = DataLoader(train_set, batch_size=len(train_set))
        val_loader = DataLoader(val_set, batch_size=len(val_set))

        if model_type == "fnn":
            fnn_model = model[0](**model_param[0]).to(device)
            trained_model = train_func(fnn_model, train_loader, val_loader, epochs, writer, output, device)
            in_sample_metrics = val_func(train_loader, trained_model, device=device)
            out_sample_metrics = val_func(val_loader, trained_model, device=device)

            y_predict = val_func(val_loader, trained_model, device=device, return_y=True)

        elif model_type == "gnn+fnn":
            gnn_model = model[0](**model_param[0]).to(device)
            fnn_model = model[1](**model_param[1]).to(device)
            gnn_trained_model, fnn_trained_model = train_func(
                gnn_model, fnn_model, train_loader, val_loader, epochs, writer, output, device=device
                )
            in_sample_metrics = val_func(train_loader, gnn_trained_model, fnn_trained_model, device=device)
            out_sample_metrics = val_func(val_loader, gnn_trained_model, fnn_trained_model, device=device)

            y_predict = val_func(val_loader, gnn_trained_model, fnn_trained_model, device=device, return_y=True)
        
        elif model_type == "linear":
            linear = model[0]
            linear_model = train_func(linear, train_loader)
            in_sample_metrics = val_func(linear_model, train_loader)
            out_sample_metrics = val_func(linear_model, val_loader)

            y_predict = val_func(linear_model, val_loader, return_y=True)

        else:
            raise ValueError("Wrong model type")
        
        return in_sample_metrics, out_sample_metrics, y_predict