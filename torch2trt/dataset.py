import torch


__all__ = [
    'DatasetRecorder',
    'Dataset',
    'ListDataset',
    'TensorBatchDataset'
]


class DatasetRecorder(object):

    def __init__(self, dataset, module):
        self.dataset = dataset
        self.module = module
        self.handle = None

    def __enter__(self, *args, **kwargs):

        if self.handle is not None:
            raise RuntimeError('DatasetRecorder is already active.')

        self.handle = self.module.register_forward_pre_hook(self._callback)

        return self

    def __exit__(self, *args, **kwargs):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _callback(self, module, input):
        self.dataset.insert(input)


class Dataset(object):

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def insert(self, item):
        raise NotImplementedError

    def record(self, module):
        return DatasetRecorder(self, module)


class ListDataset(Dataset):

    def __init__(self, items=None):
        if items is None:
            items = []
        self.items = [t for t in items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def insert(self, item):
        self.items.append(item)


class TensorBatchDataset(Dataset):

    def __init__(self, tensors=None):
        self.tensors = tensors

    def __len__(self):
        if self.tensors is None:
            return 0
        else:
            return len(self.tensors[0])

    def __getitem__(self, idx):
        if self.tensors is None:
            raise IndexError('Dataset is empty.')
        return [t[idx:idx+1] for t in self.tensors]

    def insert(self, tensors):
        if self.tensors is None:
            self.tensors = tensors
        else:
            if len(self.tensors) != len(tensors):
                raise ValueError('Number of inserted tensors does not match the number of tensors in the current dataset.')

            self.tensors = tuple([
                torch.cat((self.tensors[index], tensors[index]), dim=0) 
                for index in range(len(tensors))
            ])