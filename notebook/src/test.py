import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
from dataset import VectDataset


def test(model, dataloader, criterion, weights=None):
    """test function"""
    # if weights is not None:
    #TODO: load weights

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Running on {device}")
    model.to(device=device)
    losses = []
    print(device)
    for ind_batch, (batch_vector, batch_labels) in enumerate(dataloader):
        batch_vectors = batch_vector.to(device=device)
        batch_labels = batch_labels.to(device=device)
        with torch.no_grad():
            output, hidden_state = model(batch_vectors, None)
            loss = criterion(torch.squeeze(output[:, -1]), batch_labels.type(torch.float))
        losses.append(loss.cpu().numpy())

    global_loss = np.mean(np.asarray(losses))
    print("MSELoss test loss = {:0.3f}".format(global_loss))



