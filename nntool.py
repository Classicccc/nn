import numpy as np
import math

alpha = 0.5

def activationFunc(x,a):
    if x < 0:
        return 1 - 1 / (1 + math.exp(x*a*2))
    else:
        return 1 / (1 + math.exp(-x*a*2))

def stepForward(inputData, weights, layers):

    c = []
    for i in layers:
        c.append(np.zeros(i))
    neurons = np.array(c)

    neurons[0] = np.copy(inputData)
    i = 1
    while i < len(neurons):
        for k, neuron in enumerate(neurons[i]):
            j = 0
            amount = 0
            while j < len(neurons[i-1]):
                amount = amount + neurons[i-1][j] * weights[i-1][j][k]
                j = j + 1
            neurons[i][k] = activationFunc(amount, alpha)
        i = i + 1
    
    return neurons
