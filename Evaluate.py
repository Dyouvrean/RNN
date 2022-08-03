import torch
import numpy as np


def evaluate(model, loader_valX, loader_valY, device,loss_func):
    model.eval()
    loss_val_total = 0
    predictions, true_vals = [], []
    with torch.no_grad():
        for X_batch, Y_batch in zip(loader_valX, loader_valY):

            X_batch.to(device)
            Y_batch = Y_batch.cuda()


            outputs = model(X_batch.float())

            loss = loss_func(torch.max(outputs, 1)[0], Y_batch)
            logits = outputs
            loss_val_total += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = Y_batch.cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)

    loss_val_avg = loss_val_total/len(loader_valX)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    return loss_val_avg, predictions, true_vals
