import tensorflow as tf
import numpy as np
import os
import pickle
import logging
import math

class DNNClassifier(object):
    def __init__(self, input_size, hidden_layers, output_size, learn_rate = 0.01, model_dir = 'model'):
        self.__logger = logging.getLogger("DNNClassifier")
        self.__session = tf.Session()

        #input layer
        self.__logger.debug('add input layer, size: %d', input_size)
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
            self.__logger.debug('add hidden layer #%d size:%d stddev:%f', i+1, hidden_layer_size, stddev)

        #output layer
        self.__logger.debug('add output layer, size:%d', output_size)
        weight = tf.Variable(tf.zeros([last_layer_size, output_size]))
        bias = tf.Variable(tf.zeros([output_size]))
        self.__output = tf.nn.softmax(tf.matmul(last_layer, weight) + bias)

        #train op
        self.__test = tf.placeholder(tf.float32, [None, output_size])
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.__test * tf.log(self.__output), reduction_indices=[1]))
        self.__train_step = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy)

        #test model accuracy
        correct_prediction = tf.equal(tf.argmax(self.__test, 1), tf.argmax(self.__output, 1))
        self.__accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #all variables defined, initial them
        init = tf.global_variables_initializer()
        self.__session.run(init)

        #Saver must create after variables you want to restore/save was created
        self.__saver = tf.train.Saver(max_to_keep=10)

        self.__model_dir = model_dir
        if os.path.exists(self.__model_dir):
            model_path = tf.train.latest_checkpoint(model_dir)
            self.__saver.restore(self.__session, model_path)

    '''save model'''
    def save(self, step=None):
        if not os.path.exists(self.__model_dir):
            os.mkdir(self.__model_dir, 0755)
        self.__saver.save(self.__session, self.__model_dir + "/model", global_step=step)
        return self.__model_dir

    def _add_hidden_layer(self, x, x_size, layer_size, stddev=0.1):
        weight = tf.Variable(tf.truncated_normal([x_size, layer_size], stddev=0.1))
        bias = tf.Variable(tf.zeros([layer_size]))
        hidden_layer = tf.nn.relu(tf.matmul(x, weight) + bias)
        return hidden_layer

    '''test model with x_test and y_test, see if accuracy is satisfied'''
    def accuracy(self, x_test, y_test):
        return self.__session.run(self.__accuracy, feed_dict={
            self.__input: x_test, 
            self.__test: y_test, 
            self.__keep_prob: 1
        })

    '''train model with x_train and y_train'''
    def fit(self, x_train, y_train, keep_prob = 0.5):
        self.__session.run(self.__train_step, feed_dict={
            self.__input: x_train, 
            self.__test: y_train,
            self.__keep_prob: keep_prob
        })

    '''get predict result'''
    def predict(self, x, labels):
        result = self.__session.run(self.__output, feed_dict={self.__input: x, self.__keep_prob: 1})
        result = np.argmax(result, 1)
        
        out = []
        for idx in result:
            label = labels[idx]
            out.append(label)
        return out

