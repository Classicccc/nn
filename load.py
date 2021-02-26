import numpy as np
import nntool

inputData = np.array([1,1,1])
weights = np.load("bestWeights/11.5278(3431).npy", allow_pickle=True)
layers = [3, 4, 3, 1]

neurons = nntool.stepForward(inputData, weights, layers)

print(neurons)
print(inputData[0]+ inputData[1]*2 + inputData[2]*3)