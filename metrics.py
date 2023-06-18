import os
import torch

from tensorboardX import SummaryWriter
from datetime import datetime
from torch_geometric.loader.dataloader import DataLoader
from tqdm.notebook import trange


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


def cross_validation(dataset, num_folds, model_type, model, model_param, train_func, val_func, epochs, output=False, logs=False):

    total_size = len(dataset)
    fold_size = int(total_size / num_folds)

    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    metrics = {"in-sample loss": [], "in-sample R2": [], "out-of-sample loss": [], "out-of-sample R2": []}
    for i in trange(num_folds):

        writer = SummaryWriter("./logs/" + datetime_now + "_" + str(i)) if logs else None

        in_sample_metrics, out_sample_metrics = validation_step(
            i, fold_size, total_size, dataset, model_type, model, model_param, 
            train_func, val_func, epochs, writer, output
            )
        
        if out_sample_metrics["valid_r2"] < 0:
            print(f"Bad initialized weights. The training in {i} fold is repeating...")
            in_sample_metrics, out_sample_metrics = validation_step(
                i, fold_size, total_size, dataset, model_type, model, model_param, 
                train_func, val_func, epochs, writer, output
                )

        metrics["in-sample loss"].append(round(float(in_sample_metrics["valid_loss"]), 4))
        metrics["in-sample R2"].append(round(float(in_sample_metrics["valid_r2"]), 4))
        metrics["out-of-sample loss"].append(round(float(out_sample_metrics["valid_loss"]), 4))
        metrics["out-of-sample R2"].append(round(float(out_sample_metrics["valid_r2"]), 4))

        break

    return metrics

    
def validation_step(i, fold_size, total_size, dataset, model_type, model, model_param, 
                    train_func, val_func, epochs, writer, output):
            
        trll = 0
        trlr = i * fold_size
        vall = trlr
        valr = i * fold_size + fold_size
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
            fnn_model = model[0](**model_param)
            trained_model = train_func(fnn_model, train_loader, val_loader, epochs, writer, output)
            in_sample_metrics = val_func(train_loader, trained_model)
            out_sample_metrics = val_func(val_loader, trained_model)

        elif model_type == "gnn+fnn":
            gnn_model = model[0](**model_param[0])
            fnn_model = model[1](**model_param[1])
            gnn_trained_model, fnn_trained_model = train_func(gnn_model, fnn_model, train_loader, val_loader, epochs, writer, output)
            in_sample_metrics = val_func(train_loader, gnn_trained_model, fnn_trained_model)
            out_sample_metrics = val_func(val_loader, gnn_trained_model, fnn_trained_model)
        
        else:
            raise ValueError("Wrong model type")
        
        return in_sample_metrics, out_sample_metrics