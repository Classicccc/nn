import numpy as np
import nntool
from PIL import Image, ImageFilter
import networkx as nx
import matplotlib.pyplot as plt

# inputData = np.array([9,9,9])
# weights = np.load("bestWeights/1.3049.npy", allow_pickle=True)
# layers = [3, 10, 1]

# neurons = nntool.stepForward(inputData, weights, layers)
# print(neurons)
# print(inputData[0]+ inputData[1]*2 + inputData[2]*3)

img = Image.open("img.jpg")
img2 = Image.open("img2.jpg")
size = (128, 128)
graphSize = 10
layers = [size[0]*size[1], 100, graphSize ** 2]
# weights = nntool.generateWeights(layers)
# np.save('generatedWeights/128x128.100.100', weights, True)
weights = np.load("generatedWeights/128x128.100.100.npy", allow_pickle=True)

graph1 = nntool.generateRandomGraph(img, size, weights, layers)
graph2 = nntool.generateRandomGraph(img2, size, weights, layers)

print(graph1)
print(graph2)
print(nntool.getVectorNodes(graph1))
print(nntool.getVectorNodes(graph2))

print(nntool.qualityOfRandom(graph1, graph2))

graphs = []
graphs.append(graph1)
graphs.append(graph2)
graphs.append(graph1)
graphs.append(graph2)
graphs.append(graph1)
graphs.append(graph2)
graphs.append(graph1)
graphs.append(graph2)
graphs = np.array(graphs)
testGraph1 = graphs[0]
print(testGraph1)
graphs = (nntool.generateEvolutionGraphs(graphs, 10000, fitnessVector=[2,8,8,8,8,8,8,8,8,8]))

testGraph2 = graphs[0]
print(nntool.getVectorNodes(graphs[0]))
print(testGraph2)
print(nntool.qualityOfRandom(testGraph1, testGraph2))

G = nx.Graph(graphs[0])
nx.draw(G)
plt.show()
