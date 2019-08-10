import os


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

