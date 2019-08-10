import torch.optim as optim
import torch
import random
from torchvision import transforms
from misc import *
from DigitDataloader import load_data
import argparse
from DigitModel import load_model
from MCD_DA_Function import discrepancy


def train():
    log_interval = 20
    max_accu = 0
    len_dataloader = min(len(SourceTrainDataLoader), len(TargetTrainDataLoader))
    for epoch in range(n_epoch):
        data_src_iter, data_tar_iter = iter(SourceTrainDataLoader), iter(TargetTrainDataLoader)
        n_train_correct = 0
        i = 1
        while i < len_dataloader+1:
            i = i+1
            netG.train()
            netC1.train()
            netC2.train()
            reset_grad()

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
            reset_grad()

            data_target = data_tar_iter.next()
            t_img, t_label = data_target[0], data_target[1].long().to(device)
            if category == 'stom':
                t_img = transform_mnist_to_3_channel_32x32(t_img)
            t_img = t_img.to(device)

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
            reset_grad()

            for t in range(num_k):
                feat_t = netG(t_img)
                output_t1 = netC1(feat_t)
                output_t2 = netC2(feat_t)
                loss_dis = discrepancy(output_t1, output_t2)
                loss_dis.backward()
                optimizerG.step()
                reset_grad()

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
            for idx, sample_val in enumerate(SourceTestDataLoader, 0):
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

            acc_1 = float(n_correct_1) / len(SourceTestDataLoader.dataset) * 100
            acc_2 = float(n_correct_2) / len(SourceTestDataLoader.dataset) * 100
            acc_en = float(n_correct_ensemble) / len(SourceTestDataLoader.dataset) * 100
            disp_str = 'Epoch:[%d/%d], source_acc:%f, val_acc1:%f, val_acc2:%f, val_acc:%f' % (epoch, n_epoch, acct, acc_1, acc_2, acc_en)
            print(disp_str)
            validLogger.write(disp_str + '\n')
            validLogger.flush()
            if max_accu < acc_en:
                max_epoch = epoch
                max_accu = acc_en
                torch.save(netG.state_dict(), './%s/%s/netG_epoch_%d.pth' % (exp, category, epoch))

        if epoch % 1 == 0:
            print('*' * 5 + ' Testing ' + '*' * 5)
            n_correct_1 = 0
            n_correct_2 = 0
            n_correct_ensemble = 0
            for it, sample_t in enumerate(TargetTestDataLoader, 0):
                img_test, label_test = sample_t[0], sample_t[1]
                if category == 'stom':
                    img_test = transform_mnist_to_3_channel_32x32(img_test)
                img_test = img_test.to(device)
                label_test = label_test.long().to(device)

                netG.eval()
                netC1.eval()
                netC2.eval()
                with torch.no_grad():
                    feat = netG(img_test)
                    output1 = netC1(feat)
                    output2 = netC2(feat)
                    output_ensemble = output1 + output2
                    pred = output1.max(1)[1]
                    n_correct_1 += (pred == label_test).sum().item()
                    pred = output2.max(1)[1]
                    n_correct_2 += (pred == label_test).sum().item()
                    pred = output_ensemble.max(1)[1]
                    n_correct_ensemble += (pred == label_test).sum().item()

            acc_1 = float(n_correct_1) / len(TargetTestDataLoader.dataset) * 100
            acc_2 = float(n_correct_2) / len(TargetTestDataLoader.dataset) * 100
            acc_en = float(n_correct_ensemble) / len(TargetTestDataLoader.dataset) * 100
            disp_str = 'Epoch:[%d/%d], source_acc:%f, test_acc1:%f, test_acc2:%f, test_acc:%f' % (epoch, n_epoch, acct, acc_1, acc_2, acc_en)
            print(disp_str)
            targetLogger.write(disp_str + '\n')
            targetLogger.flush()

    disp_str = 'max_epoch:%d, max_test_acc:%f' % (max_epoch, max_accu)
    validLogger.write(disp_str + '\n')
    validLogger.flush()


def reset_grad():
    optimizerG.zero_grad()
    optimizerC1.zero_grad()
    optimizerC2.zero_grad()


def transform_mnist_to_3_channel_32x32(img):
    img_arr = torch.chunk(img, img.size()[0], 0)
    img = None
    for tensor in img_arr:
        tensor = tensor.squeeze(0)
        tensor = transform(tensor)
        tensor = tensor.unsqueeze(0)
        if img is None:
            img = tensor
        else:
            img = torch.cat([img, tensor], 0)
    return img


if __name__ == '__main__':
    makedir()
    parser = argparse.ArgumentParser(description='MCD_DA')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--category', type=str, default='utom', choices=['utom', 'mtou', 'stom'], help="datasets transfer")
    parser.add_argument('--batch_size', type=int, default=128, help="batch_size")
    parser.add_argument('--n_epoch', type=int, default=300, help="batch_size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    args = parser.parse_args()
    batch_size = args.batch_size
    n_epoch = args.n_epoch
    lrG = args.lr
    torch.cuda.set_device(int(args.gpu_id))
    device = torch.device('cuda:'+args.gpu_id if torch.cuda.is_available() else 'cpu')
    category = args.category

    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    source, target = analysis_dir(category)
    SourceTrainDataLoader, SourceTestDataLoader = load_data(source, batch_size)
    TargetTrainDataLoader, TargetTestDataLoader = load_data(target, batch_size)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Pad(padding=2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    num_k = 4

    netG, netC1, netC2 = load_model(category)
    netG, netC1, netC2 = netG.to(device), netC1.to(device), netC2.to(device)
    optimizerG = optim.Adam(netG.parameters(), lr=lrG)
    optimizerC1 = optim.Adam(netC1.parameters(), lr=lrG)
    optimizerC2 = optim.Adam(netC2.parameters(), lr=lrG)
    loss_class = torch.nn.CrossEntropyLoss()

    exp = 'check'
    trainLogger = open('./%s/%s/train.log' % (exp, category), 'w')
    validLogger = open('./%s/%s/valid.log' % (exp, category), 'w')
    targetLogger = open('./%s/%s/target.log' % (exp, category), 'w')
    train()
    trainLogger.close()
    validLogger.close()
    targetLogger.close()
