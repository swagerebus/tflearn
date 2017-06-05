# TFLearn

tensorflow的API不太直观, tensorflow.learn又过于重型

因此对常用的一些model做封装, 方便使用


## Examples

### 1. DNNClassifier

使用多层感知机, 识别MNIST手写数据:

```python
import os
import sys
import logging

module_path = os.path.join(os.getcwd(), "../..")
sys.path.append(module_path)

from tflearn import DNNClassifier
from tensorflow.examples.tutorials.mnist import input_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s', stream=sys.stdout)

# 读取MNIST数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 创建一个多层感知机, 含一个隐含层, 神经元节点数量为300, 学习速度为0.001
model = DNNClassifier(784, [300], 10, learn_rate=0.001, model_dir="dnn")

# 迭代学习3000次
for step in xrange(0, 3000):
    step += 1
    x_train, y_train = mnist.train.next_batch(100)
    # 防止过拟合
    model.fit(x_train, y_train, keep_prob=0.5)
    if step % 100 == 0:
        acc = model.accuracy(mnist.test.images, mnist.test.labels)
        logging.info('step:%d accuracy:%.4f%%', step, acc)

# 保存模型
model.save()
```

迭代多次后, 准确率能达到97%:

```
2017-06-05 18:01:11,985 INFO root: step:100 accuracy:0.8934%
2017-06-05 18:01:12,791 INFO root: step:200 accuracy:0.9085%
2017-06-05 18:01:13,600 INFO root: step:300 accuracy:0.9211%
...
2017-06-05 18:01:34,668 INFO root: step:2800 accuracy:0.9736%
2017-06-05 18:01:35,459 INFO root: step:2900 accuracy:0.9744%
2017-06-05 18:01:36,247 INFO root: step:3000 accuracy:0.9746%
```