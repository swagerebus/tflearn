import os
import sys
import logging

module_path = os.path.join(os.getcwd(), "../..")
sys.path.append(module_path)

from tflearn import DNNClassifier
from tensorflow.examples.tutorials.mnist import input_data

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s: %(message)s', stream=sys.stdout)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
model = DNNClassifier(784, [300], 10, learn_rate=0.001)

for step in xrange(0, 3000):
    step += 1
    x_train, y_train = mnist.train.next_batch(100)
    model.fit(x_train, y_train, keep_prob=0.5)
    if step % 100 == 0:
        acc = model.accuracy(mnist.test.images, mnist.test.labels)
        logging.info('step:%d accuracy:%.4f%%', step, acc)