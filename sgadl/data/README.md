# How the data pipeline works

The data pipeline of each machine learning framework is often quite complex. This repo is no exclusion.
I'll try my best to document it here.

To get from a dataset on disk to a batch, the following steps are taken:

1. Load entries
2. SGDataset
3. Split batch
4. Prepare model inputs

### Load Entries

The dataset requires a list of annotation data, called `entries`. This is a list of dictionaries that contain information about where the images are and what annotations exist.

Depending on the dataset, different sets of entries are loaded, but all are turned into the same format. Having all entries in memory

### SGDataset

The SGDataset is the PyTorch dataset used for all underlying data. By changing the entries, SGDataset can support all kinds of datasets. For the correct format of `entries`, refer to the docstring in `SGDataset.__init__`.

The SGDataset returns data on an image-level basis. That means running `data[1055]` returns data for the 1055th image in the entries list.

### Split Batch

To fully utilise the GPU, we try to maximize how much data is put on the GPU. Therefore, data from `SGDataset` is split such that it fits on the GPU.

### Prepare Model Inputs

The split data, needs to be in a format that is suitable for `model.forward`. That's what the last step is good for.
