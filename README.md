### 深度迁移学习算法对比

在统一框架（pytorch）， 统一输入数据，尽量统一网络结构的情况下复现了5篇深度迁移学习论文中的算法。在Small Image和office-31数据集下进行性能的对比。复现论文如下：

* Baseline

* Domain-Adversarial Training of Neural Networks
* Self-ensembling for visual domain adaptation
* Maximum Classifier Discrepancy for Unsupervised Domain Adaptation
* Multi-Adversarial Domain Adaptation

`尽量统一网络结构是指MCD_DA这篇中office-31数据集上的分类器层中用了三层全连接，其余都相同`

#### 统一输入数据

* 数字数据集按以下方式进行预处理并输入

|   迁移任务    | 图像尺寸 | Batch Size | Epochs |
| :-----------: | :------: | :--------: | :----: |
| USPS to MNIST | 28x28x1  |    128     |  300   |
| MNIST to USPS | 28x28x1  |    128     |  300   |
| SVHN to MNIST | 32x32x3  |    128     |  300   |

* Office-31数据集

| 数据类别 |                           图像增强                           | Batch Size | Epochs  |
| :------: | :----------------------------------------------------------: | :--------: | :-----: |
|  训练集  | 先resize到256x256x3<br>再随机截取224x224x3<br>再随机的水平翻转 |     32     | 300-500 |
|  测试集  |                    直接resize到224╳224╳3                     |     32     | 300-500 |

说明：SVDA 这篇因为使用了很多trick的技巧，只是用这样简单的增强达不到论文的效果，所以按照作者给出的方式将图像resize到了160并padding 16，再对图像添加随机的噪声，并把batch_size设置为了56。训练了25000个Iteration。其余实现都按照上方数据进行输入，300-500 epoch是各个任务图片数量不一致，所以设置不同的epoch将迁移任务总Iteration数设置在7500左右。

#### 复现结果

表中数据为所有epoch中目标域测试集的最大准确率。

* 数字数据集

|          | M->U | U->M | S->M | Avg  |
| :------: | :--: | :--: | :--: | :--: |
| Baseline | 89.0 | 87.6 | 79.2 | 85.3 |
|   DANN   | 91.7 | 95.9 | 90.7 | 92.8 |
|   SVDA   | 95.2 | 99.1 | 99.4 | 97.9 |
|  MCD_DA  | 96.9 | 98.8 | 90.5 | 95.4 |
|   MADA   | 91.9 | 95.8 | 94.6 | 94.1 |

![](./transfer_task_digit.png)

* office31数据集

|          | A->W | D->W | W->D  | A->D | D->A | W->A | Avg  |
| :------: | :--: | :--: | :---: | :--: | :--: | :--: | :--: |
| Resnet50 | 78.2 | 96.9 | 99.6  | 82.1 | 65.6 | 65.5 | 81.3 |
|   DANN   | 79.7 | 97.9 | 100.0 | 83.3 | 66.4 | 65.9 | 82.2 |
|   SVDA   | 83.8 | 96.9 | 100.0 | 82.5 | 69.7 | 69.2 | 83.7 |
|  MCD_DA  | 90.4 | 98.9 | 100.0 | 89.0 | 72.6 | 72.2 | 87.2 |
|   MADA   | 89.2 | 98.0 | 100.0 | 87.3 | 67.5 | 65.6 | 84.6 |

![](./transfer_task_office31.png)

