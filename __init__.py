import tensorflow as tf
import numpy as np
import os
import pickle
import logging
import math

optimizer_map = {
    'adam': tf.train.AdamOptimizer,
}

def Optimizer(name, learn_rate):
    name = name.lower()
    opt = optimizer_map[name]
    return opt(learn_rate)

def cross_entropy(y_, y):
    return tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

class Model(object):
    '''base model'''

    def __init__(self, name, model_dir = 'model'):
        self.__session = tf.Session()
        self.logger = logging.getLogger(name)
        self.__model_dir = model_dir
        self.__saver = None
        
    def init_variables(self):
        init = tf.global_variables_initializer()
        self.__session.run(init)

        #Saver must create after variables you want to restore/save was created
        self.__saver = tf.train.Saver(max_to_keep=10)
        if os.path.exists(self.__model_dir):
            self.restore(self.__model_dir)

    def restore(self, model_dir):
        '''restore session from checkpoint, return checkpoint path'''
        model_path = tf.train.latest_checkpoint(model_dir)
        if model_path:
            self.__saver.restore(self.__session, model_path)
        return model_path

    def save(self, step=None):
        '''save model'''
        if not os.path.exists(self.__model_dir):
            os.mkdir(self.__model_dir, 0755)
        self.__saver.save(self.__session, self.__model_dir + "/model", global_step=step)
        return self.__model_dir

    def run(self, ops, feed_dict):
        '''run single or multi operations'''
        return self.__session.run(ops, feed_dict=feed_dict)

class LinearClassifer(Model):
    pass

class DNNClassifier(Model):
    '''multi layer prodece classifier'''
    def __init__(self, layers, optimizer = None, model_dir = 'model'):
        Model.__init__(self, "DNNClassifier", model_dir)

        input_size = layers[0]
        output_size = layers[-1]
        hidden_layers = layers[1:-1]

        #input layer
        self.logger.debug('add input layer, size: %d', input_size)
        self.__input = tf.placeholder(tf.float32, [None, input_size])
        self.__keep_prob = tf.placeholder(tf.float32)

        #hidden layers
        last_layer_size = input_size
        last_layer = self.__input

        for i in range(0, len(hidden_layers)):
            hidden_layer_size = hidden_layers[i]
            stddev = 1.0 / math.sqrt(float(last_layer_size))
            #stddev = 0.1
            hidden_layer = self._add_hidden_layer(last_layer, last_layer_size, hidden_layer_size, stddev)

            if i == 0:
                dropout = tf.nn.dropout(hidden_layer, self.__keep_prob)
                last_layer = dropout
            else:
                last_layer = hidden_layer

            last_layer_size = hidden_layer_size
            self.logger.debug('add hidden layer #%d size:%d stddev:%f', i+1, hidden_layer_size, stddev)

        #output layer
        self.logger.debug('add output layer, size:%d', output_size)
        weight = tf.Variable(tf.zeros([last_layer_size, output_size]))
        bias = tf.Variable(tf.zeros([output_size]))
        self.__output = tf.nn.softmax(tf.matmul(last_layer, weight) + bias)

        #train op
        self.__test = tf.placeholder(tf.float32, [None, output_size])
        loss = cross_entropy(self.__test, self.__output)
        if optimizer == None:
            optimizer = tf.train.AdamOptimizer(0.01)
        self.__train_step = optimizer.minimize(loss)

        #test model accuracy
        correct_prediction = tf.equal(tf.argmax(self.__test, 1), tf.argmax(self.__output, 1))
        self.__accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #init all variables
        self.init_variables()

    def _add_hidden_layer(self, x, x_size, layer_size, stddev=0.1):
        weight = tf.Variable(tf.truncated_normal([x_size, layer_size], stddev=stddev))
        bias = tf.Variable(tf.zeros([layer_size]))
        hidden_layer = tf.nn.relu(tf.matmul(x, weight) + bias)
        return hidden_layer


    def accuracy(self, x_test, y_test):
        '''test model with x_test and y_test, see if accuracy is satisfied'''
        return self.run(self.__accuracy, feed_dict={
            self.__input: x_test, 
            self.__test: y_test, 
            self.__keep_prob: 1
        })

    def fit(self, x_train, y_train, keep_prob = 0.5):
        '''train model with x_train and y_train'''
        self.run(self.__train_step, feed_dict={
            self.__input: x_train, 
            self.__test: y_train,
            self.__keep_prob: keep_prob
        })

    def predict(self, x, labels = None):
        '''get predict result'''
        result = self.run(self.__output, feed_dict={self.__input: x, self.__keep_prob: 1})
        #select the most likely label index
        result = np.argmax(result, 1)

        if labels == None:
            return result

        out = []
        for idx in result:
            label = labels[idx]
            out.append(label)
        return out

class CNNClassifier(Model):
    def __init__(self, model_dir = "model"):
        Model.__init__("CNNClassifier", model_dir)
        self.init_variables()

    def fit(self, x_train, y_train, keep_prob = 0.5):
        pass

    def predict(self, x, labels = None):
        pass