import torch.optim as optim
import torch
import random
from torchvision import transforms
from misc import *
from DigitDataloader import load_data
import argparse
from model import load_model
import numpy as np


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
            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            netG.train()
            data_source = data_src_iter.next()
            optimizerG.zero_grad()
            s_img, s_label = data_source[0].to(device), data_source[1].long().to(device)
            domain_label = torch.zeros(batch_size).long().to(device)
            train_class_output, domain_output = netG(input_data=s_img, alpha=alpha)
            err_s_label = loss_class(train_class_output, s_label)
            err_s_domain = loss_domain(domain_output, domain_label)

            # Training model using target data
            data_target = data_tar_iter.next()
            t_img, t_label = data_target[0], data_target[1].long().to(device)
            if category == 'stom':
                t_img = transform_mnist_to_3_channel_32x32(t_img)
            t_img = t_img.to(device)
            domain_label = torch.ones(batch_size).long().to(device)
            _, domain_output = netG(input_data=t_img, alpha=alpha)
            err_t_domain = loss_domain(domain_output, domain_label)
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

        acct = float(n_train_correct) / (len_dataloader*batch_size) * 100
        if epoch % 1 == 0:
            print('*' * 5 + ' Validating ' + '*' * 5)
            n_correct = 0
            alpha = 0
            for idx, sample_val in enumerate(SourceTestDataLoader, 0):
                img_val, label_val = sample_val[0], sample_val[1]
                img_val = img_val.to(device)
                label_val = label_val.long().to(device)
                netG.eval()
                with torch.no_grad():
                    class_output, _ = netG(img_val, alpha)
                    pred = class_output.max(1)[1]
                    n_correct += (pred == label_val).sum().item()

            accu = float(n_correct) / len(SourceTestDataLoader.dataset) * 100
            disp_str = 'Epoch:[%d/%d], source_acc:%f, test_acc:%f' % (epoch, n_epoch, acct, accu)
            print(disp_str)
            validLogger.write(disp_str + '\n')
            validLogger.flush()
            if max_accu < accu:
                max_epoch = epoch
                max_accu = accu
                torch.save(netG.state_dict(), './%s/%s/netG_epoch_%d.pth' % (exp, category, epoch))

        if epoch % 1 == 0:
            print('*' * 5 + ' Testing ' + '*' * 5)
            n_target_correct = 0
            alpha = 0
            for it, sample_t in enumerate(TargetTestDataLoader, 0):
                img_test, label_test = sample_t[0], sample_t[1]
                if category == 'stom':
                    img_test = transform_mnist_to_3_channel_32x32(img_test)
                img_test = img_test.to(device)
                label_test = label_test.long().to(device)
                netG.eval()
                with torch.no_grad():
                    class_output, _ = netG(img_test, alpha)
                    pred = class_output.max(1)[1]
                    n_target_correct += (pred == label_test).sum().item()

            acctrg = float(n_target_correct) / len(TargetTestDataLoader.dataset) * 100
            disp_str = 'Epoch:[%d/%d], source_test_acc:%f, target_acc:%f' % (epoch, n_epoch, accu, acctrg)
            print(disp_str)
            targetLogger.write(disp_str + '\n')
            targetLogger.flush()

    disp_str = 'max_epoch:%d, max_test_acc:%f' % (max_epoch, max_accu)
    validLogger.write(disp_str + '\n')
    validLogger.flush()


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
    parser = argparse.ArgumentParser(description='DANN')
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
    netG = load_model(category).to(device)
    optimizerG = optim.Adam(netG.parameters(), lr=lrG)
    loss_class = torch.nn.CrossEntropyLoss()
    loss_domain = torch.nn.CrossEntropyLoss()

    exp = 'check'
    trainLogger = open('./%s/%s/train.log' % (exp, category), 'w')
    validLogger = open('./%s/%s/valid.log' % (exp, category), 'w')
    targetLogger = open('./%s/%s/target.log' % (exp, category), 'w')
    train()
    trainLogger.close()
    validLogger.close()
    targetLogger.close()
