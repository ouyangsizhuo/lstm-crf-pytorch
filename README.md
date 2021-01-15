# LSTM-CRF in PyTorch

## 数据介绍


数据准备
```
git clone https://github.com/ouyangsizhuo/lstm-crf-pytorch.git
cd lstm-crf-pytorch
```
实验所需的所有文件都在这个文件夹中

数据来源：TAC2017adr
下载地址：https://bionlp.nlm.nih.gov/tac2017adversereactions/

（已下载好，解压train_xml.rar即可，共有101个.xml文件）
将这些.xml文件转换成.tab格式
（格式已转换好，解压tab.rar即可，其中包括train，valid，test三个文件夹）

.tab文件格式为：
```
token tag
token tag
...
```
包含两列，第一列是单词，第二列是其对应的标签

输入文件格式要求：
```
token/tag token/tag token/tag ...
token/tag token/tag token/tag ...
...
```
分别将train，valid和test数据集转换成这样的输入格式，转换后的数据存放在prepare_data文件夹中，并命名为train.txt，valid.txt，test.txt。数据准备完毕。

## 模型训练

### 模型一：LSTM-CRF

①、参数设置：（修改下列文件中对应行的代码）

parameters.py：（第13行）
```
EMBED = {"lookup": 300}
```
predict.py：（第16行）
```
load_checkpoint('model.epoch20', model)
```
train.py：（第34行）
```
num_epochs = 20
```
②、训练过程
```
python3 train.py
```
这一步会得到训练好的模型并打印出模型结构
```
python3 predict.py
```
用训练好的模型进行预测，得到结果文件test_out.tab

对结果进行评估
```
perl conlleval.pl –d $'\t' <test_out.tab | tee test_out_lstm.eval
```

### 模型二：SA-LSTM-CRF

①、参数设置：（修改下列文件中对应行的代码）

parameters.py：（第13行）
```
EMBED = {"sae": 300}
```
predict.py：（第16行）
```
load_checkpoint('model.epoch30', model)
```
train.py：（第34行）
```
num_epochs = 30
```
②、训练过程
```
python3 train.py
python3 predict.py
perl conlleval.pl –d $'\t' <test_out.tab | tee test_out_sa.eval
```

### 模型三：LSTM-SA-CRF

①、参数设置

parameters.py：（第13行）
```
EMBED = {"sae": 300}
```
predict.py：（第16行）
```
load_checkpoint('model.epoch30', model)
```
train.py：（第34行）
```
num_epochs = 30
```
model.py：去掉第八行的注释，即在第7行和第9行之间加入第八行（这三行的顺序一定不能错！）

将第38行换成第39行（注释掉第38行，去掉第39行的注释）
```
(7) self.rnn = rnn(cti_size, wti_size, num_tags)
(8) self.embed = embed(EMBED, cti_size, wti_size, HRE)
(9) self.crf = crf(num_tags)
... ...

(38) self.embed = embed(EMBED, cti_size, wti_size, HRE)
(39) self.embed = embed({"lookup": 300}, cti_size, wti_size, HRE)
```
②、训练过程
```
python3 train.py
python3 predict.py
perl conlleval.pl –d $'\t' <test_out.tab | tee test_out_lsc.eval
```

## 注意事项

①、在训练下一个模型之前需要把之前训练好的模型新建一个文件夹存放，（不要直接放在lstm-crf-pytorch这个文件夹下），然后再开始新模型的训练。

②、训练时间较久，可以选用GPU加速。
