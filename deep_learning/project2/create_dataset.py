import torch
from torch.utils.data import TensorDataset


def create_dataset(data_path, output_path=None, contrast_normalization=False, whiten=False):
    """
    Reads and optionally preprocesses the data.

    Arguments
    --------
    data_path: (String), the path to the file containing the data
    output_path: (String), the name of the file to save the preprocessed data to (optional)
    contrast_normalization: (boolean), flags whether or not to normalize the data (optional). Default (False)
    whiten: (boolean), flags whether or not to whiten the data (optional). Default (False)

    Returns
    ------
    train_ds: (TensorDataset, the examples (inputs and labels) in the training set
    val_ds: (TensorDataset), the examples (inputs and labels) in the validation set
    """
    # read the data and extract the various sets


    # apply the necessary preprocessing as described in the assignment handout.
    # You must zero-center both the training and test data
    if data_path == "image_categorization_dataset.pt":
        # do mean centering here


        # %%% DO NOT EDIT BELOW %%%% #
        if contrast_normalization:
            image_std = torch.std(data_tr[sets_tr == 1], unbiased=True)
            image_std[image_std == 0] = 1
            data_tr = data_tr / image_std
            data_te = data_te / image_std
        if whiten:
            examples, rows, cols, channels = data_tr.size()
            data_tr = data_tr.reshape(examples, -1)
            W = torch.matmul(data_tr[sets_tr == 1].T, data_tr[sets_tr == 1]) / examples
            E, V = torch.linalg.eigh(W)
            E = E.real
            V = V.real

            en = torch.sqrt(torch.mean(E).squeeze())
            M = torch.diag(en / torch.max(torch.sqrt(E.squeeze()), torch.tensor([10.0])))

            data_tr = torch.matmul(data_tr.mm(V.T), M.mm(V))
            data_tr = data_tr.reshape(examples, rows, cols, channels)

            data_te = data_te.reshape(-1, rows * cols * channels)
            data_te = torch.matmul(data_te.mm(V.T), M.mm(V))
            data_te = data_te.reshape(-1, rows, cols, channels)

        preprocessed_data = {"data_tr": data_tr, "data_te": data_te, "sets_tr": sets_tr, "label_tr": label_tr}
        if output_path:
            torch.save(preprocessed_data, output_path)

    train_ds = TensorDataset(data_tr[sets_tr == 1], label_tr[sets_tr == 1])
    val_ds = TensorDataset(data_tr[sets_tr == 2], label_tr[sets_tr == 2])

    return train_ds, val_ds

