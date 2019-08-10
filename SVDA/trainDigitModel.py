import torch.optim as optim
import torch
import random
from torchvision import transforms
from misc import *
from DigitDataloader import load_data
import argparse
from DigitModel import load_model
from SVDA_Functions import *
from torch.nn import functional as F


def train():
    log_interval = 10
    max_accu = 0
    len_dataloader = min(len(SourceTrainDataLoader), len(TargetTrainDataLoader))
    for epoch in range(n_epoch):
        data_src_iter, data_tar_iter = iter(SourceTrainDataLoader), iter(TargetTrainDataLoader)
        n_train_correct = 0
        i = 1
        while i < len_dataloader+1:
            i = i+1

            stu_optimizer.zero_grad()
            net_stu.train()
            net_tea.train()
            data_source = data_src_iter.next()
            s_img, s_label = data_source[0].to(device), data_source[1].long().to(device)
            train_class_output = net_stu(s_img)
            err_s_label = loss_class(train_class_output, s_label)

            # Training model using target data
            data_target = data_tar_iter.next()
            t_img, t_label = data_target[0], data_target[1].long().to(device)
            if category == 'stom':
                t_img = transform_mnist_to_3_channel_32x32(t_img)
            t_img = t_img.to(device)
            target_class_output_stu = net_stu(t_img)
            target_pro_output_stu = F.softmax(target_class_output_stu, dim=1)
            target_class_output_tea = net_tea(t_img)
            target_pro_output_tea = F.softmax(target_class_output_tea, dim=1)
            unsup_loss, conf_mask_count, unsup_mask_count = compute_aug_loss(target_pro_output_stu, target_pro_output_tea)
            loss_expr = err_s_label + unsup_loss * unsup_weight
            loss_expr.backward()
            stu_optimizer.step()
            tea_optimizer.step()

            pred = train_class_output.max(1)[1]
            n_train_correct += (pred == s_label).sum().item()

            if i % log_interval == 0:
                disp_str = 'Epoch:[%d/%d],Batch:[%d/%d],err_s_label:%f, err_unsup_loss:%f, all_loss:%f' % (epoch, n_epoch, i, len_dataloader, err_s_label.item(), unsup_loss.item(), loss_expr.item())
                print(disp_str)
                trainLogger.write(disp_str + '\n')
                trainLogger.flush()

        acct = float(n_train_correct) / (len_dataloader*batch_size) * 100
        if epoch % 1 == 0:
            print('*' * 5 + ' Validating ' + '*' * 5)
            n_correct_stu = 0
            n_correct_tea = 0
            for idx, sample_val in enumerate(SourceTestDataLoader, 0):
                img_val, label_val = sample_val[0], sample_val[1]
                img_val = img_val.to(device)
                label_val = label_val.long().to(device)
                net_stu.eval()
                net_tea.eval()
                with torch.no_grad():
                    class_output_stu = net_stu(img_val)
                    pred = class_output_stu.max(1)[1]
                    n_correct_stu += (pred == label_val).sum().item()
                    class_output_tea = net_tea(img_val)
                    pred = class_output_tea.max(1)[1]
                    n_correct_tea += (pred == label_val).sum().item()
            acc_stu = float(n_correct_stu) / len(SourceTestDataLoader.dataset) * 100
            acc_tea = float(n_correct_tea) / len(SourceTestDataLoader.dataset) * 100
            disp_str = 'Epoch:[%d/%d], source_train_acc:%f, source_test_stu_acc:%f, source_test_tea_acc:%f' % (epoch, n_epoch, acct, acc_stu, acc_tea)
            print(disp_str)
            validLogger.write(disp_str + '\n')
            validLogger.flush()
            if max_accu < acc_tea:
                max_epoch = epoch
                max_accu = acc_tea
                torch.save(net_tea.state_dict(), './%s/%s/netG_epoch_tea_%d.pth' % (exp, category, epoch))
                torch.save(net_stu.state_dict(), './%s/%s/netG_epoch_stu_%d.pth' % (exp, category, epoch))

        if epoch % 1 == 0:
            print('*' * 5 + ' Testing ' + '*' * 5)
            n_target_correct_stu = 0
            n_target_correct_tea = 0
            for it, sample_t in enumerate(TargetTestDataLoader, 0):
                img_test, label_test = sample_t[0], sample_t[1]
                if category == 'stom':
                    img_test = transform_mnist_to_3_channel_32x32(img_test)
                img_test = img_test.to(device)
                label_test = label_test.long().to(device)
                net_stu.eval()
                net_tea.eval()
                with torch.no_grad():
                    class_output_stu = net_stu(img_test)
                    pred = class_output_stu.max(1)[1]
                    n_target_correct_stu += (pred == label_test).sum().item()
                    class_output_tea = net_tea(img_test)
                    pred = class_output_tea.max(1)[1]
                    n_target_correct_tea += (pred == label_test).sum().item()

            acctrg_stu = float(n_target_correct_stu) / len(TargetTestDataLoader.dataset) * 100
            acctrg_tea = float(n_target_correct_tea) / len(TargetTestDataLoader.dataset) * 100
            disp_str = 'Epoch:[%d/%d], source_test_tea_acc:%f, target_stu_acc:%f, target_tea_acc:%f' % (epoch, n_epoch, acc_tea, acctrg_stu, acctrg_tea)
            print(disp_str)
            targetLogger.write(disp_str + '\n')
            targetLogger.flush()

    disp_str = 'max_epoch:%d, max_test_tea_acc:%f' % (max_epoch, max_accu)
    print(disp_str)
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
    parser = argparse.ArgumentParser(description='SVDA')
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
    teacher_alpha = 0.99
    unsup_weight = 3.0

    net_stu = load_model(category).to(device)
    net_tea = load_model(category).to(device)
    stu_optimizer = optim.Adam(net_stu.parameters(), lr=lrG)
    tea_optimizer = EMAWeightOptimizer(net_tea, net_stu, alpha=teacher_alpha)
    loss_class = torch.nn.CrossEntropyLoss()

    exp = 'check'
    trainLogger = open('./%s/%s/train.log' % (exp, category), 'w')
    validLogger = open('./%s/%s/valid.log' % (exp, category), 'w')
    targetLogger = open('./%s/%s/target.log' % (exp, category), 'w')
    train()
    trainLogger.close()
    validLogger.close()
    targetLogger.close()
