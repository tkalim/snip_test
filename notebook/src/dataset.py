import numpy as np
import torch.nn as nn
import torch
import random
import torch.utils.data as data



MAX_VECT_SIZE = 20

def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad
    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size).long()], dim=dim)


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)
        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        # pad according to max_len
        batch = list(map(lambda xy:
                    (pad_tensor(xy[0], pad=max_len, dim=self.dim), xy[1]), batch))
        # stack all
        tensor_map = list(map(lambda x : x[0], batch))
        label_map = list(map(lambda x : x[1], batch))
        xs = torch.stack(tensor_map, dim=0)
        ys = torch.LongTensor(label_map)
        return xs[:,:, None].float(), ys

    def __call__(self, batch):
        return self.pad_collate(batch)


def create_dataset(dataset_size):
    """Creates a shuffled Dataset with balanced vector sizes for the technical challenge
    returns: vectors : list of np arrays
    labels : list of matching labels
    """
    n_vect_per_size = dataset_size // 20

    vectors, labels = [], []
    for vect_size in range(MAX_VECT_SIZE):
        random_matrix = np.random.randint(-10, high=11, size=(n_vect_per_size, vect_size+1))
        norms = np.linalg.norm(random_matrix, ord=1, axis=1)
        vectors += [line for line in random_matrix]
        labels += [norm for norm in norms]

    zipped_list = list(zip(vectors, labels))
    random.shuffle(zipped_list)
    vectors, labels = zip(*zipped_list)

    return list(vectors), list(labels)


class VectDataset(data.Dataset):

    def __init__(self, dataset_size):
        self.dataset_size = dataset_size
        self.max_vect_size = MAX_VECT_SIZE
        self.vectors, self.labels = create_dataset(self.dataset_size)

    def __getitem__(self, index):
        array = torch.from_numpy(self.vectors[index])
        label = self.labels[index]
        return array, label

    def __len__(self):
        return(len(self.labels))



