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
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s', stream=sys.stdout)

from tflearn import DNNClassifier
from tensorflow.examples.tutorials.mnist import input_data

# 读取MNIST数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# 使用adam优化器, 最小化损失函数, 学习速度为0.001
optimizer = tflearn.Optimizer('adam', learn_rate=0.001)

# 定义含有一个隐含层的多层感知机
# 输入层: 784
# 隐含层: 神经元节点数量为300, 
# 输出层: 10
clf = DNNClassifier(layers=[784, 300, 10], optimizer=optimizer, model_dir="dnn")

# 定义包含2个隐含层的多层感知机: [100, 50]; 如果没有隐含层, 就是一个线性回归分类器
# model = DNNClassifier(layers=[784, 100, 50, 10], optimizer=optimizer, model_dir="dnn")

# 迭代学习3000次, 每次训练100个样本
for step in xrange(0, 3000):
    step += 1
    x_train, y_train = mnist.train.next_batch(100)
    # 防止过拟合, 随机使半数的神经元节点不工作;
    clf.fit(x_train, y_train, keep_prob=0.5)

    # 测试准确率
    if step % 100 == 0:
        acc = model.accuracy(mnist.test.images, mnist.test.labels)
        logging.info('step:%d accuracy:%.4f%%', step, acc)

# 保存模型
clf.save()

#预测结果
labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
print clf.predict(mnist.test.images, labels=labels)
```

迭代学习30w个样本, 准确率能达到97.46%:

```
2017-06-05 18:01:11,985 INFO root: step:100 accuracy:0.8934%
2017-06-05 18:01:12,791 INFO root: step:200 accuracy:0.9085%
2017-06-05 18:01:13,600 INFO root: step:300 accuracy:0.9211%
...
2017-06-05 18:01:34,668 INFO root: step:2800 accuracy:0.9736%
2017-06-05 18:01:35,459 INFO root: step:2900 accuracy:0.9744%
2017-06-05 18:01:36,247 INFO root: step:3000 accuracy:0.9746%
```