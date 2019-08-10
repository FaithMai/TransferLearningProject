from OfficeDataloader import load_office_data
import torch.optim as optim
import torch
import random
from Resnet import load_pretrain_resnet50
from misc import *
import argparse
import math
import torch.nn as nn


def train():
    log_interval = 20
    max_accu = 0
    for epoch in range(1, n_epoch + 1):
        LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / n_epoch), 0.75)
        if isinstance(netG, nn.DataParallel):
            optimizerG = optim.SGD([
                {'params': netG.module.sharedNet.parameters()},
                {'params': netG.module.cls_fc.parameters(), 'lr': LEARNING_RATE},
            ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)
        else:
            optimizerG = optim.SGD([
                {'params': netG.sharedNet.parameters()},
                {'params': netG.cls_fc.parameters(), 'lr': LEARNING_RATE},
            ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)
        len_traindata = len(SourceTrainDataLoader)
        n_train_correct = 0
        for i, sample in enumerate(SourceTrainDataLoader, 0):
            img, label = sample[0], sample[1]
            img = img.to(device)
            label = label.long().to(device)

            optimizerG.zero_grad()
            netG.train()
            with torch.enable_grad():
                class_output = netG(img)
                loss = loss_class(class_output, label)
                loss.backward()
                optimizerG.step()
            pred = class_output.max(1)[1]
            n_train_correct += (pred == label).sum().item()

            if i % log_interval == 0:
                disp_str = 'Epoch:[%d/%d],Batch:[%d/%d],loss:%f' % (epoch, n_epoch, i, len_traindata, loss.item())
                print(disp_str)
                trainLogger.write(disp_str + '\n')
                trainLogger.flush()
        acct = float(n_train_correct) / len(SourceTrainDataLoader.dataset) * 100

        if epoch % 1 == 0:
            print('*' * 5 + ' Validating ' + '*' * 5)
            n_correct = 0
            for idx, sample_val in enumerate(TargetTestDataLoader, 0):
                img_val, label_val = sample_val[0], sample_val[1]
                img_val = img_val.to(device)
                label_val = label_val.long().to(device)
                netG.eval()
                with torch.no_grad():
                    class_output = netG(img_val)
                    pred = class_output.max(1)[1]
                    n_correct += (pred == label_val).sum().item()
            accu = float(n_correct) / len(TargetTestDataLoader.dataset) * 100
            disp_str = 'Epoch:[%d/%d],source_acc:%f,target_acc:%f' % (epoch, n_epoch, acct, accu)
            print(disp_str)
            validLogger.write(disp_str + '\n')
            validLogger.flush()
            # save model
            if max_accu < accu:
                max_epoch = epoch
                max_accu = accu
                torch.save(netG.state_dict(), './%s/%s/%s/netG_epoch_%d.pth' % (exp, data_root, category, epoch))

    disp_str = 'max_epoch:%d, max_target_acc:%f' % (max_epoch, max_accu)
    validLogger.write(disp_str + '\n')
    validLogger.flush()


if __name__ == '__main__':
    makedir()
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--dataset', type=str, default='office31', choices=['office31', 'officehome'],
                        help="The dataset used")
    parser.add_argument('--category', type=str, default='atod',
                        choices=['atod', 'atow', 'dtoa', 'dtow', 'wtoa', 'wtod', 'ArtoCl', 'ArtoPr', 'ArtoRw', 'CltoAr',
                                 'CltoPr', 'CltoRw', 'PrtoAr', 'PrtoCl', 'PrtoRw', 'RwtoAr', 'RwtoCl', 'RwtoPr'],
                        help="datasets transfer")
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--n_epoch', type=int, default=300, help="n_epoch")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    args = parser.parse_args()
    batch_size = args.batch_size   # results from my experiments officehome:48, office31:32
    n_epoch = args.n_epoch
    lrG = args.lr
    devices = args.gpu_id.split(',')
    torch.cuda.set_device(int(devices[0]))
    device = torch.device('cuda:' + devices[0] if torch.cuda.is_available() else 'cpu')
    data_root = args.dataset
    category = args.category

    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    num_class = 31 if data_root == 'office31' else 65
    source, target = analysis_dir(category)

    SourceTrainDataLoader, TargetTestDataLoader = load_office_data(data_root, source, target, batch_size)

    netG = load_pretrain_resnet50(num_class)
    if len(devices) > 1:
        netG = nn.DataParallel(netG, device_ids=[int(i) for i in devices], output_device=int(devices[0]))

    netG = netG.to(device)

    l2_decay = 5e-4
    momentum = 0.9
    lr = 0.01

    loss_class = torch.nn.CrossEntropyLoss()

    exp = 'check'
    trainLogger = open('./%s/%s/%s/train.log' % (exp, data_root, category), 'w')
    validLogger = open('./%s/%s/%s/valid.log' % (exp, data_root, category), 'w')
    train()
    trainLogger.close()
    validLogger.close()
