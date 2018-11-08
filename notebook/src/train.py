import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from dataset import PadCollate, VectDataset
from model import RNNRegressor

DATASET_SIZE = 10000
LEARNING_RATE = 0.0001
CRITERION = nn.MSELoss()
BATCH_SIZE = 1000


def train(model, dataloader, epochs, criterion, model_weights=None):

    if model_weights is not None:
        model.load_state_dict(torch.load(model_weights))

    cuda = torch.cuda.is_available()
    # cuda = False
    if cuda:
        model = model.to(device="cuda")
        print("CUDA")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    hidden_state = None

    for epoch in range(epochs):
        for ind_batch, (batch_vectors, batch_labels) in enumerate(dataloader):
            if cuda:
                batch_vectors = batch_vectors.to(device="cuda")
                batch_labels = batch_labels.to(device="cuda")
            optimizer.zero_grad()

            output, hidden_state = model(batch_vectors, None)
            loss = criterion(torch.squeeze(output[:, -1]), batch_labels.type(torch.float))
            loss.backward()
            optimizer.step()

            if ind_batch % 10 == 0:
                print(
                    "[Epoch {}, Batch {}/{}]:  [Loss: {:03.2f}]".format(
                        epoch, ind_batch, len(dataloader), loss.data[0]
                    )
                )


if __name__ == "__main__":
    r_model = RNNRegressor()
    dataset = VectDataset(DATASET_SIZE)
    dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=PadCollate(dim=0))
    train(r_model, dataloader=dataloader, epochs=1000, criterion=CRITERION)
