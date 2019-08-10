from torchvision import datasets, transforms
import torch


def load_office_data(dir, train, test, batch_size):
    transform_train = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root='./dataset/'+dir+'/'+train, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

    transform_test = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root='./dataset/'+dir+'/'+test, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader, test_loader


def load_office_test_data(dir, test, batch_size):
    transform_test = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root='./dataset/'+dir+'/'+test, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
    return test_loader


if __name__ == '__main__':
    train_loader, test_loader = load_office_data('office31', 'amazon', 'amazon', batch_size=1)
    print(len(test_loader.dataset))
    for i, sample in enumerate(train_loader, 0):
        if i > 0:
            break
        print(i)
        print(sample[0].size())
        print(sample[1])
