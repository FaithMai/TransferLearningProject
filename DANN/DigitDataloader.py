from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import h5py
from PIL import Image
from scipy.io import loadmat as load
from misc import *


def load_mnist_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset_train = datasets.MNIST(
        root='./dataset/',
        train=True,
        transform=transform,
        download=True
    )
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2)

    dataset_test = datasets.MNIST(
        root='./dataset/',
        train=False,
        transform=transform,
        download=True
    )
    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2)
    return dataloader_train, dataloader_test


class USPS_data_train(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        with h5py.File('./dataset/usps/usps.h5', 'r') as hf:
            train = hf.get('train')
            # format:(7291, 256)
            self.train_samples = train.get('data')[:]
            # format:(7291,)
            self.train_labels = train.get('target')[:]

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, index):
        img = self.train_samples[index]
        img = img.reshape(16, 16)
        img = img[:, :, np.newaxis]
        if self.transform is not None:
            img = self.transform(img)
        sample = [img, self.train_labels[index]]
        return sample


class USPS_data_Test(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        with h5py.File('./dataset/usps/usps.h5', 'r') as hf:
            test = hf.get('test')
            # format:(2007, 256)
            self.test_samples = test.get('data')[:]
            # format:(2007,)
            self.test_labels = test.get('target')[:]

    def __len__(self):
        return len(self.test_labels)

    def __getitem__(self, index):
        img = self.test_samples[index]
        img = img.reshape(16, 16)
        img = img[:, :, np.newaxis]
        if self.transform is not None:
            img = self.transform(img)
        sample = [img, self.test_labels[index]]
        return sample


def load_usps_data(batch_size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((28, 28), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset_train = USPS_data_train(transform=transform)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    dataset_test = USPS_data_Test(transform=transform)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=2)
    return dataloader_train, dataloader_test


class SVHN_data_train(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        traindata = load('./dataset/svhn/train_32x32.mat')
        # format:(32, 32, 3, 73257)
        self.train_samples = traindata['X']
        # format:(73257, 1)
        self.train_labels = traindata['y']

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, index):
        img = self.train_samples[:, :, :, index]
        if self.transform is not None:
            img = self.transform(img)
        sample = [img, self.train_labels[index, 0] % 10]
        return sample


class SVHN_data_test(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        testdata = load('./dataset/svhn/test_32x32.mat')
        # format:(32, 32, 3, 26032)
        self.test_samples = testdata['X']
        # format:(26032, 1)
        self.test_labels = testdata['y']

    def __len__(self):
        return len(self.test_labels)

    def __getitem__(self, index):
        img = self.test_samples[:, :, :, index]
        if self.transform is not None:
            img = self.transform(img)
        sample = [img, self.test_labels[index, 0] % 10]
        return sample


def load_svhn_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dataset_train = SVHN_data_train(transform=transform)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    dataset_test = SVHN_data_test(transform=transform)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=2)
    return dataloader_train, dataloader_test


def load_data(dataset, batch_size):
    if dataset == 'mnist':
        return load_mnist_data(batch_size)
    elif dataset == 'usps':
        return load_usps_data(batch_size)
    elif dataset == 'svhn':
        return load_svhn_data(batch_size)


if __name__ == '__main__':
    x, y = load_svhn_data(128)
    cal_mean_and_std(x)
    # print(len(y.dataset))
    # for i, sample in enumerate(y, 0):
    #     if i > 1:
    #         break
    #     print(i)
    #     print(sample[0].size())
    #     print(sample[1])
