# encoding=utf8
# -*- coding: utf-8 -*-
from __future__ import division
import time
import pickle
import gzip
from random import randrange
from scipy import special
import numpy as np
import sys
LAYERS_NUMBER = 300
# =====================
#     Network class
# =====================


class Network:

    def __init__(self, num_hidden):
        self.input_size = 784
        self.output_size = 10
        self.num_hidden = num_hidden
        self.best = 0.
        self.same = 0

        hidden_layer = np.random.rand(self.num_hidden, self.input_size + 1) / self.num_hidden
        output_layer = np.random.rand(self.output_size, self.num_hidden + 1) / self.output_size
        self.layers = [hidden_layer, output_layer]
        self.iteration = 0.

        print('Initialization with random weight')
        print('-----')

    def train_GD(self, batchsize, training):
        start_time = time.time()
        print('Network training with ' + str(batchsize) + ' examples Gradient descent method ')
        print('Until convergence (10 iterations without improvements)')
        print('-----')
        inputs = training[0][0:batchsize]
        targets = np.zeros((batchsize, 10))
        for i in range(batchsize):
            targets[i, training[1][i]] = 1

        while self.same < 10:
            for input_vector, target_vector in zip(inputs, targets):
                self.backpropagate(input_vector, target_vector)
            # Messages and backups
            self.iteration += 1.
            accu = self.accu(TESTING)
            message = 'Iteration ' + str(int(self.iteration)).zfill(2) + \
                      ' (' + str(round(time.time() - start_time)).zfill(2) + 's) '
            message += 'Global accuracy:' + str(accu[1]).zfill(4) + '% Minimal digit accuracy:' + \
                       str(accu[0]).zfill(4) + '% (' + str(int(accu[2])) + ')'
            if accu[0] > self.best:
                self.same = 0
                self.best = accu[0]
                message += ' R'
                if accu[0] > 90:
                    self.sauv(file_name='ntMIN_' + str(accu))
                    message += 'S'
            else:
                self.same += 1
            print(message)
        print('10 Iterations without improvements.')
        print('Total duration: ' + str(round((time.time() - start_time), 2)) + 's')

    #  not working yet
    def train_SGD(self, batchsize, max, training):
        start_time = time.time()
        print('Network training with ' + str(batchsize) + ' examples Stochastic gradient descent method ')
        print('Until convergence (10 iterations without improvements)')
        print('-----')
        maxbatch = int(max/batchsize)
        iteration_batch = randrange(0, maxbatch - 1)
        inputs = training[0][iteration_batch * batchsize:(iteration_batch + 1) * batchsize]
        targets = np.zeros((batchsize, 10))
        for i in range(batchsize):
            targets[i, training[1][i]] = 1

        while self.same < 10:
            for input_vector, target_vector in zip(inputs, targets):
                self.backpropagate(input_vector, target_vector)
            # Messages and backups
            self.iteration += 1.
            accu = self.accu(TESTING)
            message = 'Iteration ' + str(int(self.iteration)).zfill(2) + \
                      ' (' + str(round(time.time() - start_time)).zfill(2) + 's) '
            message += 'Global accuracy:' + str(accu[1]).zfill(4) + '% Minimal digit accuracy:' + \
                       str(accu[0]).zfill(4) + '% (' + str(int(accu[2])) + ')'
            if accu[0] > self.best:
                self.same = 0
                self.best = accu[0]
                message += ' R'
                if accu[0] > 90:
                    self.sauv(file_name='ntMIN_' + str(accu))
                    message += 'S'
            else:
                self.same += 1
            print(message)
        print('10 Iterations without improvements.')
        print('Total duration: ' + str(round((time.time() - start_time), 2)) + 's')

    def feed_forward(self, input_vector):
        # Wyliczenie i podmiana sieci starej na nową
        outputs = []
        for layer in self.layers:
            input_with_bias = np.append(input_vector, 1)
            output = np.inner(layer, input_with_bias)
            output = special.expit(output)
            outputs.append(output)
            input_vector = output
        return outputs

    def backpropagate(self, input_vector, target):
        """Reduce error for one input vector:
        Calculating the partial derivatives for each coeff then subtracts"""
        c = 1. / (self.iteration + 10)  # Learning coefficient
        hidden_outputs, outputs = self.feed_forward(input_vector)

        # Calculation of partial derivatives for the output layer and subtraction
        output_deltas = outputs * (1 - outputs) * (outputs - target)

        # Calculation of partial derivatives for the hidden layer and subtraction
        hidden_deltas = hidden_outputs * (1 - hidden_outputs) * \
                        np.dot(np.delete(self.layers[-1], LAYERS_NUMBER, 1).T, output_deltas)
        self.layers[-1] -= c * np.outer(output_deltas, np.append(hidden_outputs, 1))
        self.layers[0] -= c * np.outer(hidden_deltas, np.append(input_vector, 1))

    def predict(self, input_vector):
        return self.feed_forward(input_vector)[-1]

    def predict_one(self, input_vector):
        return np.argmax(self.feed_forward(input_vector)[-1])

    def sauv(self, file_name=''):
        if file_name == '':
            file_name = 'nt_' + str(self.accu(TESTING)[0])
        sauvfile = self.layers
        f = open(file_name, 'wb')
        pickle.dump(sauvfile, f)
        f.close()

    def load(self, file_name):
        f = open(file_name, 'rb')
        self.layers = pickle.load(f, encoding='latin1')
        f.close()

    def accu(self, testing):
        #wyliczanie ilości "zgadniętych"
        res = np.zeros((10, 2))
        for k in range(len(testing[1])):
            if self.predict_one(testing[0][k]) == testing[1][k]:
                res[testing[1][k]] += 1
            else:
                res[testing[1][k]][1] += 1
        total = np.sum(res, axis=0)
        each = [res[k][0] / res[k][1] for k in range(len(res))]
        min_c = sorted(range(len(each)), key=lambda k: each[k])[0]
        return np.round([each[min_c] * 100, total[0] / total[1] * 100, min_c], 2)

    def testbenis(self):
        benis = [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                   1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1,
                   1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1,
                   1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0], [0, 0]], [0, 0]]
        self.printImg(benis, 0)
        print("prediction: " + str(self.predict_one(benis[0][0])))



    def manualtest(self):
        id = 0
        accu = self.accu(TESTING)
        for k in range(len(TESTING[1])):
            self.printImg(TESTING, id)
            print("prediction: " + str(self.predict_one(TESTING[0][id])))
            key = input('')
            id = id + 1
            if key == 'q':
                break
        return

    def printImg(self, el, id):
        idx: int
        for idx, val in enumerate(el[0][id]):
            if val == 0:
                sys.stdout.write(" ")
            elif 0 < val < 0.9:
                sys.stdout.write(".")
            elif val >= 0.9:
                sys.stdout.write("#")
            if (idx + 1) % 28 == 0:
                print("")
        print("it's: " + str(el[1][id]))


# =====================
#    main
# =====================

START_TIME = time.time()
ft = gzip.open('data_training', 'rb')
TRAINING = pickle.load(ft)  # 60000 examples max
ft.close()
ft = gzip.open('data_testing', 'rb')
TESTING = pickle.load(ft)  # 10000 examples max
ft.close()
print('Import duration ' + str(round((time.time() - START_TIME), 2)) + 's')
print('----')

neuralNetwork = Network(LAYERS_NUMBER)
neuralNetwork.load("ntMIN_[96.89 97.99  7.  ]")
# neuralNetwork.train_GD(600, TRAINING)
neuralNetwork.manualtest()

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)