## 论文来源

[Attention is all you need](https://arxiv.org/abs/1706.03762)

## 代码参考

[Harvard nlp](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

## 项目结构

- data `源数据目录`
- log  `日志存放目录 （每次预测产生一个 log-timestamp.txt）`
- save `模型存放目录`
- model `模型目录`
    - attention.py
    - embedding.py
    - encoder.py
    - decoder.py
    - generator.py
    - sublayer.py
    - position_wise_feedforward.py
    - transformer.py
- lib  `损失函数、优化器等存放位置`
    - criterion.py `损失函数`
    - optimizer.py `优化器`
    - loss.py `优化器 + 损失函数封装类`
- evaluate.py `预测.py`
- train.py `训练.py`
- parser.py `参数.py`
- utils.py `工具类.py`
- run.py `入口文件.py`
- desc_tokenizer.py `对芯片描述数据进行预处理，并生成与芯片描述相对应的index`
- README.md `readme`

#### 训练
`python3 run.py`

#### 对多个芯片描述进行分隔符预测，并对模型进行评价 (前提：训练过)
`python3 run.py --type evaluate`

#### 对用户输入和单一芯片描述进行分隔符预测（前提：训练过）
`python3 run.py --type predict`

## 数据描述
源数据目录下的data1,data2,……,data10分别存有5万，10万，……，50万条芯片描述语句。test_data_1000和test_data_10000作为预测数据，分别存有1000条和10000条芯片描述数据。源数据具体信息如下图所示。result.txt文件用于存储不同数量数据训练后的模型进行预测时，模型的准确率。
![image](https://github.com/Brauntt/delimiter_prediction/raw/master/IMG/1.png)
#### 训练数据生成规则
我们将描述中的12个feature进行合并，每个feature之间插入一个随机分隔符。
`Delimiter = [' ', '/', ',', ';', '-', '.']`
合并结束后的字符串作为翻译任务中的source language。
#### 分词规则
分词原则为一句描述中一旦遇到符号（包括空格），就实施一次分词操作。

`description = 'GENERAL PURPOSE INDUCTOR/39.0.'`

`tokenized_description = ['GENERAL', ' ', 'PURPOSE', ' ', 'INDUCTOR', '/', '39', '.', '0', '.']`
#### 编码原则
我们将描述中所有数字或字母组成的字符串标为'1'，出现在feature中的符号标为'2'（即认为是有意义的符号）分割feature与feature的符号标为'0'。
`description_index = ['1','2','1','2','1','0','1','0']`
生成的编码将作为此次翻译任务中的target lauguage。

## 模型评价方法
将标记为‘2’的word定义为真，即为有意义的出现在描述中的标点符号；将标记为‘0’的word定义为假，即为用于分隔符的符号，包括【'  ' , ' / ' , ' , ' , ' ; ' , ' - ' , ' . '】。

![image](https://github.com/Brauntt/delimiter_prediction/raw/master/IMG/2.png)

- 真实值为2，预测为2，TP➕1
- 真实值为2，预测为0，FN➕1
- 真实值为0，预测为0，TN➕1
- 真实值为2，预测为2，FP➕1

预测结束后，我们将每一组翻译中得到的TP、FN、TN、FP四个参数累加，得到完整预测数据集的相应参数。最终，通过计算真正率(True Positive Rate)和真负率(False Positive Rate)来评价模型预测结果，计算方法如下图所示。

![image](https://github.com/Brauntt/delimiter_prediction/raw/master/IMG/3.png)

模型存放目录中所的模型为15万训练数据得到的模型。在数据集大小为1000的测试集上进行测试，测试指标如下表所示：

Parameter | Value
----------| -----
total true positive amount  |  38917 
total false negative amount |  1896
total true negative amount  |  56655
total false positive amount |  2439
symbol within feature TPR   |  0.954
delimiter TNR               |  0.959
 
