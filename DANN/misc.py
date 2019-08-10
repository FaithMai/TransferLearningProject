import os
import numpy as np

def makedir():
    if os.path.exists('./check') is not True:
        os.makedirs('./check/mtou')
        os.makedirs('./check/utom')
        os.makedirs('./check/stom')
        os.makedirs('./check/office31/atod')
        os.makedirs('./check/office31/atow')
        os.makedirs('./check/office31/dtoa')
        os.makedirs('./check/office31/dtow')
        os.makedirs('./check/office31/wtoa')
        os.makedirs('./check/office31/wtod')
        os.makedirs('./check/officehome/ArtoCl')
        os.makedirs('./check/officehome/ArtoPr')
        os.makedirs('./check/officehome/ArtoRw')
        os.makedirs('./check/officehome/CltoAr')
        os.makedirs('./check/officehome/CltoPr')
        os.makedirs('./check/officehome/CltoRw')
        os.makedirs('./check/officehome/PrtoAr')
        os.makedirs('./check/officehome/PrtoCl')
        os.makedirs('./check/officehome/PrtoRw')
        os.makedirs('./check/officehome/RwtoAr')
        os.makedirs('./check/officehome/RwtoCl')
        os.makedirs('./check/officehome/RwtoPr')


def analysis_dir(transfer):
    dirs = transfer.split('to')
    return dir_name(dirs[0]), dir_name(dirs[1])


def dir_name(dir):
    if dir == 'u':
        return 'usps'
    elif dir == 'm':
        return 'mnist'
    elif dir == 's':
        return 'svhn'
    elif dir == 'a':
        return 'amazon'
    elif dir == 'd':
        return 'dslr'
    elif dir == 'w':
        return 'webcam'
    elif dir == 'Ar':
        return 'Art'
    elif dir == 'Cl':
        return 'Clipart'
    elif dir == 'Pr':
        return 'Product'
    elif dir == 'Rw':
        return 'Real World'


def cal_mean_and_std(data_loader):
    data_mean = []  # Mean of the dataset
    data_std0 = []  # std of dataset
    data_std1 = []  # std with ddof = 1
    for i, data in enumerate(data_loader, 0):
        # shape (batch_size, 3, height, width)
        numpy_image = data[0].numpy()

        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
        batch_std0 = np.std(numpy_image, axis=(0, 2, 3))
        batch_std1 = np.std(numpy_image, axis=(0, 2, 3), ddof=1)

        data_mean.append(batch_mean)
        data_std0.append(batch_std0)
        data_std1.append(batch_std1)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    data_mean = np.array(data_mean).mean(axis=0)
    data_std0 = np.array(data_std0).mean(axis=0)
    data_std1 = np.array(data_std1).mean(axis=0)

    print(data_mean, data_std0, data_std1)


if __name__ == '__main__':
    makedir()
