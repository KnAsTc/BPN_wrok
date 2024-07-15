# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import math

class BPN():
    def __init__(self, dataset, labels, sample, size, learning_rate = 10, iteration = 200):
        self.dataset = dataset
        self.x = dataset
        self.T = labels
        self.D = sample
        self.size = size
        self.eta = learning_rate
        self.iteration = iteration
        #self.setup()
        
    
    def set_bias(self):
        self.theta_h = np.random.random(self.size[1])   #hidden layer bias
        self.theta_o = np.random.random(self.size[2])   #output layer bias
        
    def set_weight(self):
        self.w_h = np.random.random((self.size[0] , self.size[1]))
        self.w_o = np.random.random((self.size[1] , self.size[2]))
        
    
    def train(self):
        self.set_bias()
        self.set_weight()
        net_h = np.zeros(self.size[1], dtype = float)    # hidden node temp
        net_o = np.zeros(self.size[2], dtype = float)    # output node temp
        H = np.zeros(self.size[1], dtype = float)        # hidden node value
        Y = np.zeros((self.D, self.size[2]), dtype = float)   # output node value
        delta_o = np.zeros(self.size[2], dtype = float) 
        delta_h = np.zeros(self.size[1], dtype = float)
        
        for t in range(self.iteration):  # t times iterations
            for d in range(self.D):      # D input sample
                
                #forward
                for j in range(self.size[1]):
                    net_h[j] = -self.theta_h[j]
                    for i in range(self.size[0]):
                        net_h[j] = net_h[j] + self.w_h[i, j] * self.x[d, i]
                    H[j] = 1 / (1 + math.exp(-net_h[j]))
                    
                for k in range(self.size[2]):
                    net_o[k] = -self.theta_o[k]
                    for j in range(self.size[1]):
                        net_o[k] = net_o[k] + self.w_o[j, k] * H[j]
                    Y[d, k] = 1 / (1 + math.exp(-net_o[k]))
                    
                #backward
                for k in range(self.size[2]):
                    delta_o[k] = Y[d, k] * (1 - Y[d, k]) * (self.T[d] - Y[d, k])
                for j in range(self.size[1]):
                    delta_h[j] = 0
                    for k in range(self.size[2]):
                        delta_h[j] = delta_h[j] + self.w_o[j, k] * delta_o[k]
                    delta_h[j] = H[j] * (1 - H[j]) * delta_h[j]
                
                #adjust
                for k in range(self.size[2]):
                    self.theta_o[k] = self.theta_o[k] - self.eta * delta_o[k] 
                    for j in range(self.size[1]):
                        self.w_o[j, k] = self.w_o[j, k] + self.eta * delta_o[k] * H[j]
                for j in range(self.size[1]):
                    self.theta_h[j] = self.theta_h[j] - self.eta * delta_h[j]
                    for i in range(self.size[0]):
                        self.w_h[i, j] = self.w_h[i, j] + self.eta * delta_h[j] * self.x[d, i]
                
            #print(Y)
       # return Y
    def predict(self,dataset):
        
        x = dataset
        net_h = np.zeros(self.size[1], dtype = float)    # input node value
        net_o = np.zeros(self.size[2], dtype = float)    # output node value
        H = np.zeros(self.size[1], dtype = float)       # hidden node value
        Y = np.zeros((self.D, self.size[2]), dtype = float)       # output node value
        for d in range(self.D):      # D input sample
                
                #forward
                for j in range(self.size[1]):
                    net_h[j] = -self.theta_h[j]
                    for i in range(self.size[0]):
                        net_h[j] = net_h[j] + self.w_h[i, j] * x[d, i]
                    H[j] = 1 / (1 + math.exp(-net_h[j]))
                    
                for k in range(self.size[2]):
                    net_o[k] = -self.theta_o[k]
                    for j in range(self.size[1]):
                        net_o[k] = net_o[k] + self.w_o[j, k] * H[j]
                    Y[d, k] = 1 / (1 + math.exp(-net_o[k]))
                    if(Y[d, k] >= 0.6):    Y[d, k] = 1
                    else:   Y[d, k] = 0
 
        return Y
     
