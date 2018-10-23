# -*- coding: utf-8 -*-
"""
A simple MLP to represent the XOR gate

@author: Yannick Wehr
"""

import numpy as np

INPUT = [[0,0],[0,1],[1,0],[1,1]]
OUTPUT = [[0],[1],[1],[0]]

LR = 0.1
ITERATIONS = 1000

N_IN = 2
N_HIDDEN = 10
N_OUT = 1

weights_ih = np.random.uniform(-0.1, 0.1, (N_HIDDEN, N_IN))
weights_ho = np.random.uniform(-0.1, 0.1, (N_OUT, N_HIDDEN))

bias_hidden = np.random.uniform(-1, 1, N_HIDDEN)
bias_out = np.random.uniform(-1, 1, N_OUT)

def main(weights_ih, weights_ho, bias_hidden, bias_out):
    
    for i in range (ITERATIONS):
        for g in range(len(INPUT)):
            
            #Feedforward activation
            hidden_layer = phi(np.dot(weights_ih, INPUT[g]) + bias_hidden)
            out_layer = np.dot(weights_ho, hidden_layer) + bias_out
            
            #Backpropagation
            error = OUTPUT[g] - out_layer
            delta_out = error * (-1)
            derivative = hidden_layer * (1 - hidden_layer)
            delta_hidden = derivative * np.transpose(np.dot(np.transpose(delta_out), weights_ho))
            
            #Adapting weights and biases
            weights_ho += -LR * np.outer(delta_out, hidden_layer)
            weights_ih += -LR * np.outer(delta_hidden, INPUT[g])
            
            bias_out += -LR * delta_out
            bias_hidden += -LR * delta_hidden
            
            print(out_layer)
            
        print()
    

def phi(x):
    return 1.0 / (1.0 + np.exp(-x))

main(weights_ih, weights_ho, bias_hidden, bias_out)