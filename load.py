import numpy as np
import nntool

inputData = np.array([5,5,5])
weights = np.load("bestWeights/2.1253.npy", allow_pickle=True)
neurons = nntool.stepForward(inputData, weights, 3, len(weights[1]), 1)

print(neurons)
print(inputData[0]+ inputData[1]*2 + inputData[2]*3)