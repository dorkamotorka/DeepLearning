import pandas as pd
import math
import numpy as np
import pickle

class Network(object):
    def __init__(self, sizes, optimizer="sgd", l2_reg=False):
        # weights connect two layers, each neuron in layer L is connected to every neuron in layer L+1,
        # the weights for that layer of dimensions size(L+1) X size(L)
        # the bias in each layer L is connected to each neuron in L+1, the number of weights necessary for the bias
        # in layer L is therefore size(L+1).
        # The weights are initialized with a He initializer: https://arxiv.org/pdf/1502.01852v1.pdf
        self.t = 1
        self.l2_reg = l2_reg
        self.num_layers = len(sizes) # First layer is input layer
        self.weights = [((2/sizes[i-1])**0.5)*np.random.randn(sizes[i], sizes[i-1]) for i in range(1, len(sizes))]
        self.biases = [np.zeros((x, 1)) for x in sizes[1:]] # no biases in the input layer
        self.optimizer = optimizer
        if self.optimizer == "adam":
            self.Vw = [np.zeros((sizes[i], sizes[i-1])) for i in range(1, len(sizes))] 
            self.Vb = [np.zeros((x, 1)) for x in sizes[1:]]
            self.Sw = [np.zeros((sizes[i], sizes[i-1])) for i in range(1, len(sizes))]
            self.Sb = [np.zeros((x, 1)) for x in sizes[1:]]
            self.dVw = [np.zeros((sizes[i], sizes[i-1])) for i in range(1, len(sizes))] 
            self.dVb = [np.zeros((x, 1)) for x in sizes[1:]]
            self.dSw = [np.zeros((sizes[i], sizes[i-1])) for i in range(1, len(sizes))]
            self.dSb = [np.zeros((x, 1)) for x in sizes[1:]]

    def train(self, training_data,training_class, val_data, val_class, epochs, mini_batch_size, eta, lmbda):
        # training data - numpy array of dimensions [n0 x m], where m is the number of examples in the data and
        # n0 is the number of input attributes
        # training_class - numpy array of dimensions [c x m], where c is the number of classes
        # epochs - number of passes over the dataset
        # mini_batch_size - number of examples the network uses to compute the gradient estimation

        iteration_index = 0
        eta_current = eta

        n = training_data.shape[1]
        for j in range(epochs):
            print("Epoch " + str(j))
            loss_avg = 0.0
            mini_batches = [
                (training_data[:,k:k + mini_batch_size], training_class[:,k:k+mini_batch_size])
                for k in range(0, n, mini_batch_size)]

            # Mini-batch is (input, target_output)
            # Currently each batch only one value
            mini_batches.pop(-1)
            for mini_batch in mini_batches:
                output, Zs, As = self.forward_pass(mini_batch[0])
                gw, gb = net.backward_pass(output, mini_batch[1], Zs, As)
                self.update_network(gw, gb, eta_current, len(mini_batch[0]), lmbda=lmbda)

                # Implement the learning rate schedule for Task 5
                # TODO: Should it be called exery mini-batch or only every epoch?
                eta_current = exp_learn_rate_decay(eta, iteration_index)
                iteration_index += 1

                loss = cross_entropy(mini_batch[1], output, self.l2_reg, self.weights, len(training_data), lmbda=lmbda)
                loss_avg += loss

            print("Epoch {} complete".format(j))
            print("Loss:" + str(loss_avg / len(mini_batches)))
            if j % 10 == 0:
                self.eval_network(val_data, val_class, lmbda)



    def eval_network(self, validation_data,validation_class, lmbda):
        # validation data - numpy array of dimensions [n0 x m], where m is the number of examples in the data and
        # n0 is the number of input attributes
        # validation_class - numpy array of dimensions [c x m], where c is the number of classes
        n = validation_data.shape[1]
        loss_avg = 0.0
        tp = 0.0
        for i in range(validation_data.shape[1]):
            example = np.expand_dims(validation_data[:,i],-1)
            example_class = np.expand_dims(validation_class[:,i],-1)
            example_class_num = np.argmax(validation_class[:,i], axis=0)
            output, Zs, activations = self.forward_pass(example)
            output_num = np.argmax(output, axis=0)[0]
            tp += int(example_class_num == output_num)

            loss = cross_entropy(example_class, output, self.l2_reg, self.weights, len(validation_data), lmbda=lmbda)
            loss_avg += loss
        print("Validation Loss:" + str(loss_avg / n))
        print("Classification accuracy: "+ str(tp/n))

    def update_network(self, gw, gb, eta, n, beta=0.9, gama=0.999, E=1e-7, lmbda=0.0001):
        # gw - weight gradients - list with elements of the same shape as elements in self.weights
        # gb - bias gradients - list with elements of the same shape as elements in self.biases
        # eta - learning rate
        # SGD
        if self.optimizer == "sgd":
            for i in range(len(self.weights)):
                if self.l2_reg:
                    self.weights[i] = (1-(eta*lmbda/n))*self.weights[i] - eta * gw[i]
                    self.biases[i] = self.biases[i] - eta * gb[i]
                else:
                    self.weights[i] = self.weights[i] - eta * gw[i]
                    self.biases[i] = self.biases[i] - eta * gb[i]
        elif self.optimizer == "adam":
            # update weights for each layer
            for i in range(len(self.weights)):
                self.Vw[i] = beta*self.Vw[i] + (1 - beta)*gw[i]
                self.Sw[i] = gama*self.Sw[i] + (1 - gama)*(gw[i]**2)
                self.Vb[i] = beta*self.Vb[i] + (1 - beta)*gb[i]
                self.Sb[i] = gama*self.Sb[i] + (1 - gama)*(gb[i]**2)
                self.dVw[i] = self.Vw[i]/(1-beta**self.t)
                self.dSw[i] = self.Sw[i]/(1-gama**self.t)
                self.dVb[i] = self.Vb[i]/(1-beta**self.t)
                self.dSb[i] = self.Sb[i]/(1-gama**self.t)
                if self.l2_reg:
                    self.weights[i] = (1-(eta*lmbda/n))*self.weights[i] - (eta / np.subtract(np.sqrt(self.dSw[i]), E))*self.dVw[i]
                    self.biases[i] = self.biases[i] - (eta / np.subtract(np.sqrt(self.dSb[i]), E))*self.dVb[i]
                else:
                    self.weights[i] = self.weights[i] - (eta / np.subtract(np.sqrt(self.dSw[i]), E))*self.dVw[i]
                    self.biases[i] = self.biases[i] - (eta / np.subtract(np.sqrt(self.dSb[i]), E))*self.dVb[i]
            self.t += 1
        else:
            raise ValueError('Unknown optimizer:'+self.optimizer)



    def forward_pass(self, input):
        # input - numpy array of dimensions [n0 x m], where m is the number of examples in the mini batch and
        # n0 is the number of input attributes
        Zs = []
        As = [input]
        for w,b in zip(self.weights, self.biases):
            Zs.append(np.dot(w, As[-1])+b)
            As.append(sigmoid(Zs[-1]))
        output = As[-1] = softmax(Zs[-1]) # Update last layer to use softmax activation function

        return output, Zs, As

    def backward_pass(self, output, target, Zs, activations):
        '''
        calculate errors for w and b in each layer
        '''
        gb = [np.zeros(b.shape) for b in self.biases]
        gw = [np.zeros(w.shape) for w in self.weights]

        # last layer softmax activation
        delta = softmax_dLdZ(output, target) / len(target)
        gb[-1] = delta
        gw[-1] = np.dot(delta, activations[-2].transpose()) 

        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(Zs[-l]) / len(target) # target je enakega size-a kot mini batch 
            gb[-l] = delta 
            gw[-l] = np.dot(delta, activations[-l-1].transpose())

        return gw, gb

def exp_learn_rate_decay(eta, t, k=0.00001):
    return eta * math.exp(-k*t)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / expZ.sum(axis=0, keepdims=True)

def softmax_dLdZ(output, target):
    # partial derivative of the cross entropy loss w.r.t Z at the last layer
    return output - target

def cross_entropy(y_true, y_pred, l2_reg, weights, n, epsilon=1e-12, lmbda=0.0001):
    targets = y_true.transpose()
    predictions = y_pred.transpose()
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    if l2_reg:
        ce = -np.sum(targets * np.log(predictions + 1e-9)) / N + lmbda/(2*N) * sum(np.linalg.norm(w)**2 for w in weights)
    else:
        ce = -np.sum(targets * np.log(predictions + 1e-9)) / N
    return ce

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

def load_data_cifar(train_file, test_file):
    train_dict = unpickle(train_file)
    test_dict = unpickle(test_file)
    train_data = np.array(train_dict['data']) / 255.0
    train_class = np.array(train_dict['labels'])
    train_class_one_hot = np.zeros((train_data.shape[0], 10))
    train_class_one_hot[np.arange(train_class.shape[0]), train_class] = 1.0
    test_data = np.array(test_dict['data']) / 255.0
    test_class = np.array(test_dict['labels'])
    test_class_one_hot = np.zeros((test_class.shape[0], 10))
    test_class_one_hot[np.arange(test_class.shape[0]), test_class] = 1.0
    return train_data.transpose(), train_class_one_hot.transpose(), test_data.transpose(), test_class_one_hot.transpose()

if __name__ == "__main__":
    train_file = "./data/train_data.pckl"
    test_file = "./data/test_data.pckl"
    train_data, train_class, test_data, test_class = load_data_cifar(train_file, test_file)
    val_pct = 0.1
    val_size = int(len(train_data) * val_pct)
    val_data = train_data[..., :val_size]
    val_class = train_class[..., :val_size]
    train_data = train_data[..., val_size:]
    train_class = train_class[..., val_size:]
    # The Network takes as input a list of the numbers of neurons at each layer. The first layer has to match the
    # number of input attributes from the data, and the last layer has to match the number of output classes
    lmbda = 0.0001
    eta = 0.0001
    net = Network([train_data.shape[0], 250, 150, 10], optimizer="adam", l2_reg=True)
    net.train(train_data,train_class, val_data, val_class, 40, 64, eta, lmbda)
    net.eval_network(test_data, test_class, lmbda)
