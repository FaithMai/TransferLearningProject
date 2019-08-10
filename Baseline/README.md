### Baseline实现

1. 数字集的迁移，使用Self-Ensembling for Visual Domain Adaptation论文附录D中的网络结构（将fc层改为三层）对源域数据进行训练，然后直接预测目标域数据准确率。
2. office31和officehome数据集迁移，使用ImageNet预训练的Resnet50作为基础结构，将最后全连接分类层分类数改为31类，对网络进行微调。在源域上训练数据后直接在目标域数据进行测试。

#### 实现结果

* 数字数据集

其中Test_max值是所有epoch中在源域上的测试集达到最大准确率时，目标域测试集的准确率。Target_max是所有epoch中目标域测试集的最大准确率

|            | M->U | U->M | S->M | Avg  |
| :--------: | :--: | :--: | :--: | :--: |
|  Test_max  | 83.9 | 83.2 | 75.1 | 80.7 |
| Target_max | 89.0 | 87.6 | 79.2 | 85.3 |

* office31数据集

|             | A->W | D->W | W->D | A->D | D->A | W->A | Avg  |
| :---------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| Source only | 78.2 | 96.9 | 99.6 | 82.1 | 65.6 | 65.5 | 81.3 |

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

训练的参数可以自己指定，默认数字数据集batch_size为128，office数据集batch_size为32。训练epoch为300。

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

office数据集除了以上参数还需要指定是office31还是officehome。此外，可以同时指定多块GPU同时进行训练，中间用`,`隔开。

```
Amazon->Webcam
python3 trainResnet50.py --gpu_id 0 --dataset office31 --category atow

Dslr->Webcam
python3 trainResnet50.py --gpu_id 0 --dataset office31 --category dtow --batch_size 32 --n_epoch 300

Webcam->Dslr
python3 trainResnet50.py --gpu_id 0 --dataset office31 --category wtod --batch_size 32 --n_epoch 300

Ar->Cl
python3 trainResnet50.py --gpu_id 0,1 --dataset officehome --category ArtoCl --batch_size 32 --n_epoch 300

Ar->Pr
python3 trainResnet50.py --gpu_id 0,1 --dataset officehome --category ArtoPr --batch_size 32 --n_epoch 300

Ar->Rw
python3 trainResnet50.py --gpu_id 0,1 --dataset officehome --category ArtoRw --batch_size 32 --n_epoch 300
```

#### 结果查看

训练的结果可以在运行过程中实时查看，也可以在保存的文件中查看。

* 数字数据集

数字数据集共产生3个结果文件`train.log`，保存训练损失等数据。`valid.log`，保存源域测试集准确率数据。`target.log`，保存目标域测试集准确率数据。文件写入到`check`目录对应的目录下。

* office数据集

office数据集共产生2个结果文件`train.log`，保存训练损失等数据。`valid.log`，保存源域准确率和目标域准确率。

check目录生成结果

```
check
|-- mtou
|   |-- train.log
|   |-- valid.log
|   `-- target.log
|-- office31
|   |-- atod
|   |-- atow
|   |-- dtoa
|   |-- dtow
|   |-- wtoa
|   `-- wtod
|-- officehome
|   |-- ArtoCl
|   |-- ArtoPr
|   |-- ArtoRw
|   |-- CltoAr
|   |-- CltoPr
|   |-- CltoRw
|   |-- PrtoAr
|   |-- PrtoCl
|   |-- PrtoRw
|   |-- RwtoAr
|   |-- RwtoCl
|   `-- RwtoPr
|-- stom
`-- utom
```

