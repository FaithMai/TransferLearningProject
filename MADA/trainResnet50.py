from OfficeDataloader import load_office_data, load_office_test_data
import torch.optim as optim
import torch
import random
from misc import *
import argparse
from model import load_MADA
import torch.nn as nn
import math
from MADA_function import logits_BCE
import torch.nn.functional as F


def train():
    log_interval = 10
    max_accu = 0
    alpha = 0
    len_dataloader = min(len(SourceTrainDataLoader), len(TargetTrainDataLoader))
    for epoch in range(1, n_epoch+1):
        LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / n_epoch), 0.75)
        if isinstance(netG, nn.DataParallel):
            optimizerG = optim.SGD([
                {'params': netG.module.feature.parameters()},
                {'params': netG.module.class_classifier.parameters(), 'lr': LEARNING_RATE},
                {'params': netG.module.domain_classifiers.parameters(), 'lr': LEARNING_RATE},
            ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)
        else:
            optimizerG = optim.SGD([
                {'params': netG.feature.parameters()},
                {'params': netG.class_classifier.parameters(), 'lr': LEARNING_RATE},
                {'params': netG.domain_classifiers.parameters(), 'lr': LEARNING_RATE},
            ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)
        data_src_iter, data_tar_iter = iter(SourceTrainDataLoader), iter(TargetTrainDataLoader)
        n_train_correct = 0
        i = 1
        while i < len_dataloader+1:
            i = i+1
            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            netG.train()
            data_source = data_src_iter.next()
            optimizerG.zero_grad()
            s_img, s_label = data_source[0].to(device), data_source[1].long().to(device)
            train_class_output, domain_output = netG(input_data=s_img, alpha=alpha)
            domain_label_train = torch.zeros(s_img.size(0)).float().to(device)
            err_s_label = loss_class(train_class_output, s_label)
            err_s_domain = loss_domain(domain_output, domain_label_train)

            data_target = data_tar_iter.next()
            t_img, t_label = data_target[0].to(device), data_target[1].long().to(device)
            domain_label_target = torch.ones(t_img.size(0)).float().to(device)
            _, domain_output = netG(t_img, alpha)
            err_t_domain = loss_domain(domain_output, domain_label_target)
            err = err_s_label + err_t_domain + err_s_domain
            err.backward()
            optimizerG.step()

            pred = train_class_output.max(1)[1]
            n_train_correct += (pred == s_label).sum().item()

            if i % log_interval == 0:
                disp_str = 'Epoch:[%d/%d],Batch:[%d/%d],err_s_label:%f, err_s_domain:%f, err_t_domain:%f' % (epoch, n_epoch, i, len_dataloader, err_s_label.item(), err_s_domain.item(), err_t_domain.item())
                print(disp_str)
                trainLogger.write(disp_str + '\n')
                trainLogger.flush()

        print_alpha = alpha
        acct = float(n_train_correct) / (len_dataloader*batch_size) * 100
        if epoch % 1 == 0:
            print('*' * 5 + ' Validating ' + '*' * 5)
            n_correct = 0
            alpha = 0
            for idx, sample_val in enumerate(TargetTestDataLoader, 0):
                img_val, label_val = sample_val[0], sample_val[1]
                img_val = img_val.to(device)
                label_val = label_val.long().to(device)
                netG.eval()
                with torch.no_grad():
                    test_class_output, _ = netG(img_val, alpha)
                    pred = test_class_output.max(1)[1]
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
        print('Current Alpha:', print_alpha, '  Current Lr:', LEARNING_RATE, '  Current tar_max:', max_accu)
    disp_str = 'max_epoch:%d, max_target_acc:%f' % (max_epoch, max_accu)
    validLogger.write(disp_str + '\n')
    validLogger.flush()


if __name__ == '__main__':
    makedir()
    parser = argparse.ArgumentParser(description='MADA')
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

    SourceTrainDataLoader, TargetTrainDataLoader = load_office_data(data_root, source, target, batch_size)
    TargetTestDataLoader = load_office_test_data(data_root, target, batch_size)

    netG = load_MADA(num_class)
    if len(devices) > 1:
        netG = nn.DataParallel(netG, device_ids=[int(i) for i in devices], output_device=int(devices[0]))

    netG = netG.to(device)

    loss_class = torch.nn.CrossEntropyLoss()
    loss_domain = logits_BCE

    l2_decay = 5e-4
    momentum = 0.9
    lr = 0.01

    exp = 'check'
    trainLogger = open('./%s/%s/%s/train.log' % (exp, data_root, category), 'w')
    validLogger = open('./%s/%s/%s/valid.log' % (exp, data_root, category), 'w')
    train()
    trainLogger.close()
    validLogger.close()
