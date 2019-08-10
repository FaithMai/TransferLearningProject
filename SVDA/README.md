### SVDA复现

1. 数字集的迁移，使用Self-Ensembling for Visual Domain Adaptation论文附录D中的网络结构（将fc层改为三层）对源域数据进行训练，然后直接预测目标域数据准确率。
2. office31和officehome数据集迁移，使用ImageNet预训练的Resnet50作为基础结构，将最后全连接分类层分类数改为31类。

#### 实现结果

* 数字数据集

其中Test_max值是所有epoch中在源域上的测试集达到最大准确率时，目标域测试集的准确率。Target_max是所有epoch中目标域测试集的最大准确率

|            | M->U | U->M | S->M | Avg  |
| :--------: | :--: | :--: | :--: | :--: |
|  Test_max  | 94.5 | 98.9 | 96.5 | 96.6 |
| Target_max | 95.2 | 99.1 | 99.4 | 97.9 |

* office31数据集

|             | A->W | D->W | W->D  | A->D | D->A | W->A | Avg  |
| :---------: | :--: | :--: | :---: | :--: | :--: | :--: | :--: |
| Source only | 83.8 | 96.9 | 100.0 | 82.5 | 69.7 | 69.2 | 83.7 |

#### 数据集准备

为了避免数据格式不同，更好的使用实现的代码，可以使用我所使用的数据集，下载好以后放置到dataset文件夹中。

* usps数据集[下载](https://www.kaggle.com/bistaumanga/usps-dataset)

* svhn数据集[下载](http://ufldl.stanford.edu/housenumbers/)

* office31数据集[下载](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view)

* officehome数据集[下载](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view)

最终数据集目录结构如下：

```
dataset
|-- MNIST
|   |-- processed
|   `-- raw
|-- office31
|   |-- amazon
|   |-- dslr
|   `-- webcam
|-- officehome
|   |-- Art
|   |-- Clipart
|   |-- Product
|   `-- Real World
|-- svhn
|   |-- test_32x32.mat
|   `-- train_32x32.mat
`-- usps
    `-- usps.h5
```

#### 训练

训练的参数可以自己指定，默认数字数据集batch_size为128。

* 数字数据集

```
USPS->MNIST
python3 trainDigitModel.py --gpu_id 0 --category utom

MNIST->USPS
python3 trainDigitModel.py --gpu_id 0 --category mtou --batch_size 128 --n_epoch 300

SVHN->MNIST
python3 trainDigitModel.py --gpu_id 0 --category stom --batch_size 128 --n_epoch 300
```

* office数据集

office数据集采用作者代码进行复现，下方指令可以完成我的复现过程。

```
python trainResnet50.py --category=atod --log_file=results_office/res_office_atod_resnet50_run${2}.txt --result_file=results_office/history_office_atod_resnet50_run${2}.h5 --model_file=results_office/model_office_atod_resnet50_run${2}.pkl --gpu_id=0

python trainResnet50.py --category=atow --log_file=results_office/res_office_atow_resnet50_run${2}.txt --result_file=results_office/history_office_atow_resnet50_run${2}.h5 --model_file=results_office/model_office_atow_resnet50_run${2}.pkl --gpu_id=0

python trainResnet50.py --category=dtoa --log_file=results_office/res_office_dtoa_resnet50_run${2}.txt --result_file=results_office/history_office_dtoa_resnet50_run${2}.h5 --model_file=results_office/model_office_dtoa_resnet50_run${2}.pkl --gpu_id=0

python trainResnet50.py --category=dtow --log_file=results_office/res_office_dtow_resnet50_run${2}.txt --result_file=results_office/history_office_dtow_resnet50_run${2}.h5 --model_file=results_office/model_office_dtow_resnet50_run${2}.pkl --gpu_id=0

python trainResnet50.py --category=wtoa --log_file=results_office/res_office_wtoa_resnet50_run${2}.txt --result_file=results_office/history_office_wtoa_resnet50_run${2}.h5 --model_file=results_office/model_office_wtoa_resnet50_run${2}.pkl --gpu_id=0

python trainResnet50.py --category=wtod --log_file=results_office/res_office_wtod_resnet50_run${2}.txt --result_file=results_office/history_office_wtod_resnet50_run${2}.h5 --model_file=results_office/model_office_wtod_resnet50_run${2}.pkl --gpu_id=0
```

#### 结果查看

训练的结果可以在运行过程中实时查看，也可以在保存的文件中查看。

* 数字数据集

数字数据集共产生3个结果文件`train.log`，保存训练损失等数据。`valid.log`，保存源域测试集准确率数据。`target.log`，保存目标域测试集准确率数据。文件写入到`check`目录对应的目录下。

```
check
|-- mtou
|   |-- train.log
|   |-- valid.log
|   `-- target.log
|-- stom
`-- utom
```

* office数据集

office数据集共产生3个文件结果，`res_office_atow_resnet50_run.txt`，`history_office_atow_resnet50_run`是保存的运行结果，`model_office_atow_resnet50_run.pkl`是最终的模型。

```
results_office
|-- res_office_atow_resnet50_run.txt
|-- history_office_atow_resnet50_run.h5
`-- model_office_atow_resnet50_run.pkl
```


