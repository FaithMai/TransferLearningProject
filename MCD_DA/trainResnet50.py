from OfficeDataloader import load_office_data, load_office_test_data
import torch.optim as optim
import torch
import random
from misc import *
import argparse
from Resnet import load_pretrain_resnet50
import torch.nn as nn
import math
from MCD_DA_Function import discrepancy


def train():
    log_interval = 10
    max_accu = 0
    len_dataloader = min(len(SourceTrainDataLoader), len(TargetTrainDataLoader))
    for epoch in range(1, n_epoch+1):
        LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / n_epoch), 0.75)
        if isinstance(netG, nn.DataParallel):
            optimizerG = optim.SGD(netG.module.parameters(), lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)
        else:
            optimizerG = optim.SGD(netG.parameters(), lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)
        optimizerC1 = optim.SGD(netC1.parameters(), lr=LEARNING_RATE, momentum=momentum, weight_decay=l2_decay)
        optimizerC2 = optim.SGD(netC2.parameters(), lr=LEARNING_RATE, momentum=momentum, weight_decay=l2_decay)

        data_src_iter, data_tar_iter = iter(SourceTrainDataLoader), iter(TargetTrainDataLoader)
        n_train_correct = 0
        i = 1
        while i < len_dataloader+1:
            i = i+1
            netG.train()
            netC1.train()
            netC2.train()
            optimizerG.zero_grad()
            optimizerC1.zero_grad()
            optimizerC2.zero_grad()

            data_source = data_src_iter.next()
            s_img, s_label = data_source[0].to(device), data_source[1].long().to(device)
            feat_s = netG(s_img)
            output_s1 = netC1(feat_s)
            output_s2 = netC2(feat_s)
            loss_s1 = loss_class(output_s1, s_label)
            loss_s2 = loss_class(output_s2, s_label)
            loss_s = loss_s1 + loss_s2
            loss_s.backward()
            optimizerG.step()
            optimizerC1.step()
            optimizerC2.step()
            optimizerG.zero_grad()
            optimizerC1.zero_grad()
            optimizerC2.zero_grad()

            data_target = data_tar_iter.next()
            t_img, t_label = data_target[0].to(device), data_target[1].long().to(device)
            feat_s = netG(s_img)
            output_s1 = netC1(feat_s)
            output_s2 = netC2(feat_s)
            feat_t = netG(t_img)
            output_t1 = netC1(feat_t)
            output_t2 = netC2(feat_t)
            loss_s1 = loss_class(output_s1, s_label)
            loss_s2 = loss_class(output_s2, s_label)
            loss_s = loss_s1 + loss_s2
            loss_dis = discrepancy(output_t1, output_t2)
            loss = loss_s - loss_dis
            loss.backward()
            optimizerC1.step()
            optimizerC2.step()
            optimizerG.zero_grad()
            optimizerC1.zero_grad()
            optimizerC2.zero_grad()

            for t in range(num_k):
                feat_t = netG(t_img)
                output_t1 = netC1(feat_t)
                output_t2 = netC2(feat_t)
                loss_dis = discrepancy(output_t1, output_t2)
                loss_dis.backward()
                optimizerG.step()
                optimizerG.zero_grad()
                optimizerC1.zero_grad()
                optimizerC2.zero_grad()

            pred = (output_s1+output_s2).max(1)[1]
            n_train_correct += (pred == s_label).sum().item()

            if i % log_interval == 0:
                disp_str = 'Epoch:[%d/%d],Batch:[%d/%d],loss_class:%f, loss_dis:%f' % (epoch, n_epoch, i, len_dataloader, loss.item(), loss_dis.item())
                print(disp_str)
                trainLogger.write(disp_str + '\n')
                trainLogger.flush()

        acct = float(n_train_correct) / (len_dataloader*batch_size) * 100
        if epoch % 1 == 0:
            print('*' * 5 + ' Validating ' + '*' * 5)
            n_correct_1 = 0
            n_correct_2 = 0
            n_correct_ensemble = 0
            for idx, sample_val in enumerate(TargetTestDataLoader, 0):
                img_val, label_val = sample_val[0], sample_val[1]
                img_val = img_val.to(device)
                label_val = label_val.long().to(device)
                netG.eval()
                netC1.eval()
                netC2.eval()
                with torch.no_grad():
                    feat = netG(img_val)
                    output1 = netC1(feat)
                    output2 = netC2(feat)
                    output_ensemble = output1 + output2
                    pred = output1.max(1)[1]
                    n_correct_1 += (pred == label_val).sum().item()
                    pred = output2.max(1)[1]
                    n_correct_2 += (pred == label_val).sum().item()
                    pred = output_ensemble.max(1)[1]
                    n_correct_ensemble += (pred == label_val).sum().item()

            acc_1 = float(n_correct_1) / len(TargetTestDataLoader.dataset) * 100
            acc_2 = float(n_correct_2) / len(TargetTestDataLoader.dataset) * 100
            acc_en = float(n_correct_ensemble) / len(TargetTestDataLoader.dataset) * 100
            disp_str = 'Epoch:[%d/%d], source_acc:%f, val_acc1:%f, val_acc2:%f, val_acc:%f' % (epoch, n_epoch, acct, acc_1, acc_2, acc_en)
            print(disp_str)
            validLogger.write(disp_str + '\n')
            validLogger.flush()
            # save model
            if max_accu < acc_en:
                max_epoch = epoch
                max_accu = acc_en
                torch.save(netG.state_dict(), './%s/%s/%s/netG_epoch_%d.pth' % (exp, data_root, category, epoch))
        print('Current Lr:', LEARNING_RATE, '  Current tar_max:', max_accu)
    disp_str = 'max_epoch:%d, max_target_acc:%f' % (max_epoch, max_accu)
    print(disp_str)
    validLogger.write(disp_str + '\n')
    validLogger.flush()


if __name__ == '__main__':
    makedir()
    parser = argparse.ArgumentParser(description='DANN')
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

    netG, netC1, netC2 = load_pretrain_resnet50(num_class)
    if len(devices) > 1:
        netG = nn.DataParallel(netG, device_ids=[int(i) for i in devices], output_device=int(devices[0]))

    netG, netC1, netC2 = netG.to(device), netC1.to(device), netC2.to(device)

    l2_decay = 5e-4
    momentum = 0.9
    lr = 0.01

    num_k = 4
    loss_class = torch.nn.CrossEntropyLoss()

    exp = 'check'
    trainLogger = open('./%s/%s/%s/train.log' % (exp, data_root, category), 'w')
    validLogger = open('./%s/%s/%s/valid.log' % (exp, data_root, category), 'w')
    train()
    trainLogger.close()
    validLogger.close()
