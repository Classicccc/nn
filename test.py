import numpy as np
import math
import random
import sympy
import signal, sys
# array = np.zeros((5,5,5), float)

# print(array)

# for element in array:
#     print(element+1)

# i=0
# print(i)
# while i < 3:
#     print(array[i])
#     i +=1
# # print("Hello World+223", variable, type(variable[0]))

# f(x,y,z)= x+2*y+3*z    3 нейрона на входном слое. 1 нейрон на выходном. 1 скрытый слой - 5 нейронов.
# 20 весовых значений

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def signal_handler(signal, frame):
    np.save('bestWeights/'+str(minError), bestWeights, True)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def activationFunc(x,a):
    if x < 0:
        return 1 - 1 / (1 + math.exp(x*a*2))
    else:
        return 1 / (1 + math.exp(-x*a*2))

def stepForward(inputData, weights):
    neurons = np.array((np.zeros([numberInputNeurons]), np.zeros([numberHiddenNeurons]), np.zeros([numberOutputNeurons])))
    neurons[0] = np.copy(inputData)
    i = 1
    while i < 3:
        for k, neuron in enumerate(neurons[i]):
            j = 0
            amount = 0
            while j < len(neurons[i-1]):
                amount = amount + neurons[i-1][j] * weights[i-1][j][k]
                j = j + 1
            neurons[i][k] = activationFunc(amount, alpha)
        i = i + 1
    
    return neurons


def stepBackPropagation(neurons, weights, target):
    # total = 1/2(target - out)^2 + 1/2(target2 - out2)
    # dtotal/dout = -(target - out)
    # dout/dnet = out*(1-out)
    # dnet/dwi = prevOut = neoruns[i]

    out = neurons[len(neurons)-1][0]
    dtotal = out - target
    dout = out*(1-out)

    newWeights = np.copy(weights)

    # for prelast layer

    i = 0
    for j in weights[len(weights)-1]:
        for weight in j:
            dnet = neurons[len(neurons)-2][i]
            correction = dtotal * dout * dnet
            weight = weight - alpha * correction
            newWeights[len(weights)-1][i][0] = weight
            i = i + 1

    # for other layers
    i = len(weights)-2
    while i > -1:
        for j, weight in enumerate(weights[i]):
            for k, value in enumerate(weights[i][j]):
                dout2 = neurons[i+1][k] * (1 - neurons[i+1][k])
                dtotal2 =  (dtotal*dout) * weights[i][j][k]
                dnet2 = neurons[i][j]
                correction2 = dtotal2 * dout2 * dnet2
                newWeight = weights[i][j][k] - alpha * correction2
                newWeights[i][j][k] = newWeight
        i = i - 1

    return newWeights

# f(x,y,z)= x+2*y+3*z  

numberInputNeurons = 3
numberHiddenNeurons = 10
numberOutputNeurons = 1
alpha = 0.5

inputData = np.array([-1,-1,-1])
outputData = np.array([-1])
i = 0
j = 0
k = 0
while i<10:
    j = 0
    while j<10:
        k = 0
        while k<10:
            inputData = np.vstack((inputData, np.array([i, j ,k])))
            outputData = np.append(outputData, (i+2*j+3*k)/100)
            k = k + 1
        j = j + 1
    i = i + 1

# print(np.vstack((np.array([i, j, k]), np.array([i, j, k]))))
print((inputData))
print((outputData))
numberWeights = numberHiddenNeurons*numberInputNeurons + numberHiddenNeurons*numberOutputNeurons

weights = np.array((np.zeros([numberInputNeurons,numberHiddenNeurons]), np.zeros([numberHiddenNeurons,numberOutputNeurons])))

for k, nvm in enumerate(weights):
    for i, nvm in enumerate(weights[k]):
        for j, nvm in enumerate(weights[k][i]):
            weights[k][i][j] = random.uniform(-1, 1)

i = 1
neurons = stepForward(np.array([6,6,6]), weights)
before = neurons[2][0]
j = 1
error = 0
errorup = 0
minError = 1000
# 50000 optimum
while True:
    neurons = stepForward(inputData[i], weights)
    # print(i, "Neurons----------------- \n" , neurons)
    # print(i, "Wieghts----------------- \n" , weights)
    weights = stepBackPropagation(neurons, weights, outputData[i])
    if error < abs(neurons[2][0] - outputData[i]):
        errorup = errorup + 1
    else:
        errorup = 0
    if errorup > 50000 and alpha > 0.01:
        alpha = alpha - alpha/5

    error = error + abs(neurons[2][0] - outputData[i])
    # print(i, "New Wieghts----------------- \n" , weights)

    j = j + 1
    i = i + 1
    if (i>1000):
        i = 0
        print("error = ", error)
        if error < minError:
            minError = error
            bestWeights = np.copy(weights)
        if (error / (1000) < 0.001):
            break
        error = 0
        print(j)
        print("Min Error = ", minError)


print("Iterations: ", j)
neurons = stepForward(np.array([6,6,6]), weights)
print(666, "After Neurons-----------------", 6+2*6+3*6,"  \n" , neurons[2][0])
neurons = stepForward(np.array([1,2,3]), weights)
print(123, "After Neurons-----------------", 1+2*2+3*3,"  \n" , neurons[2][0])
neurons = stepForward(np.array([0,4,3]), weights)
print(43, "After Neurons-----------------", 0+2*4+3*3,"  \n" , neurons[2][0])
neurons = stepForward(np.array([9,9,9]), weights)
print(999, "After Neurons-----------------", 9+2*9+3*9,"  \n" , neurons[2][0])
neurons = stepForward(np.array([4,2,6]), weights)
print(426, "After Neurons-----------------", 4+2*2+3*6," \n" , neurons[2][0])
neurons = stepForward(np.array([11,11,11]), weights)
print(151515, "After Neurons-----------------", 11+2*11+3*11," \n" , neurons[2][0])
# print(sympy.diff(0.5*(sympy.symbols("y")-sympy.symbols("x"))**2+0.5*(sympy.symbols("y2")-sympy.symbols("x2"))**2, sympy.symbols("x")))
