import torch
import torch.utils.data as data
import numpy as np



class Dataset(data.Dataset):
    def __init__(self, path, transform=None, normalize=False):
        self.path = path
        self.data = np.load(path+"/kat_256.npy")
        self.label = np.load(path+"/katlabel_256.npy")
        self.transforms = transform
        self.normalize = normalize

    def __getitem__(self, i):
        if i < 0 or i >= self.label.shape[0]:
            raise IndexError("Index out of bounds")
        d = self.data[i,:,:]
        label = self.label[i]
        #class_label, domain_label = get_label(self.label[i])
        if self.normalize:
            x_mean = d.mean()
            x_max = d.max()
            d = (d - x_mean) / x_max
        if self.transforms is not None:
            d = self.transforms(d)
        #return torch.tensor(d, dtype=torch.float32), torch.tensor(class_label, dtype=torch.long), torch.tensor(domain_label, dtype=torch.long)
        return torch.tensor(d, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return self.label.shape[0]


def get_data_loader(batch_size, transforms, normalize, target=True, shuffle=True, train=True):
    """build the data loader of the data set"""
    if target and train:
        s = 'target_train'
        dataset = Dataset('./data/condition4/train', transforms, normalize)
    elif target and train is False:
        s = 'target_test'
        dataset = Dataset('./data/condition4/test', transforms, normalize)
    elif not target and train:
        s = 'source_train'
        dataset = Dataset('./data/condition1/train', transforms, normalize)
    else:
        s = 'source_test'
        dataset = Dataset('./data/condition1/test', transforms, normalize)

    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print('got {0} data_loader'.format(s))
    return data_loader


def get_label(input_data):
    if isinstance(input_data, np.int32):
        return input_data, 0
    elif isinstance(input_data, np.ndarray) and input_data.shape == (2,):
        return input_data[0], input_data[1]
    else:
        raise ValueError("Input must be an integer or a (2, 1) shaped numpy array")