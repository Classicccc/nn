import numpy as np
import math
import random
from PIL import Image, ImageFilter
from functools import reduce

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

def generateWeights(layers):
    c = []
    for i in range(len(layers)-1):
        c.append(np.zeros([layers[i], layers[i+1]]))

    weights = np.array(c)

    for k, nvm in enumerate(weights):
        print('.', end="")
        for i, nvm in enumerate(weights[k]):
            for j, nvm in enumerate(weights[k][i]):
                weights[k][i][j] = random.uniform(-1, 1)
    print(".")
    return weights

def graphConnectivity(graph):
    visited = np.zeros(len(graph))

    def dfs(u, visited):
        visitedVertices = 1
        visited[u] = True
        for i, value in enumerate(graph[u]):
            if value == 1 and visited[i]==0:
                visitedVertices += dfs(i, visited)
        return visitedVertices

    connect = False
    for i in range(len(graph)):
        visited = np.zeros(len(graph))
        if dfs(i, visited) == len(graph):
            connect = True
            break

    return connect

def generateRandomGraph(img, size, weights, layers):
    graphSize = int(layers[len(layers)-1] ** 0.5)
    convImg = img.resize((size[0], size[1])).convert("L")
    inputData = list(convImg.getdata())

    outputData = stepForward(inputData, weights, layers)[len(layers)-1]
    graph = np.zeros([graphSize, graphSize])
    for i in range(len(outputData)):
        if outputData[i] >= 0.5:
            graph[i // graphSize][i % graphSize] = 1

    return graph


def getVectorNodes(graph):
    vector = []
    for i in range(len(graph)):
        count = 0
        for j in range(len(graph[i])):
            if (graph[i][j] == 1 or graph[j][i] == 1) and (i != j):
                count += 1
        vector.append(count)
    return vector

def graphIdentity(graph1, graph2):
    value = 0
    for i in range(len(graph1)):
        value += reduce(lambda x,y: x + y, map(lambda x,y: 0 if (x == y) else 1, graph1[i], graph2[i]))
    return value

def qualityOfRandom(graph1, graph2):
    vector1 = getVectorNodes(graph1)
    vector2 = getVectorNodes(graph2)
    vectorDifference = reduce(lambda x,y: x + y, map(lambda x,y: abs(x-y), vector1, vector2)) / len(graph1) ** 2
    identity = graphIdentity(graph1, graph2) / len(graph1) ** 2
    return [vectorDifference, identity]

def fitnessDenseGraph(graphs, dense):
    priorities = []
    fitnessResult = []
    for i in graphs:
        fitnessResult.append(reduce(lambda x,y: x+y, i))

    print(fitnessResult)

    for i in fitnessResult:
        if dense:
            m = reduce(lambda x,y: x if (x > y) else y, fitnessResult)
        else:
            m = reduce(lambda x,y: x if (x < y) else y, fitnessResult)
        priorities.append(fitnessResult.index(m))
        if dense:
            fitnessResult[fitnessResult.index(m)] = -1
        else:
            fitnessResult[fitnessResult.index(m)] = 999999
    return priorities

def fitnessVectorGraph(graphs, vector):
    priorities = []
    amount = []
    vectors = []
    listGraphs = []
    array = np.array(graphs)
    for i, value in enumerate(array):
        listGraphs.append(value.reshape(int(len(value)**0.5), int(len(value)**0.5)).tolist())
        
    for i,value in enumerate(listGraphs):
        vectors.append(getVectorNodes(value))
        amount.append(reduce(lambda x,y: x + y, map(lambda x,y: abs(x-y), vectors[i], vector)))
    
    for i in amount:
        m = reduce(lambda x,y: x if (x < y) else y, amount)
        priorities.append(amount.index(m))
        amount[amount.index(m)] = 999999


    return priorities

def fitnessTreeGraph(graphs):

    priorities = []
    amount = []
    connected = []
    notConnected = []
    for i in graphs:
        amount.append(reduce(lambda x,y: x+y, i))
    array = np.array(graphs)
    for i, value in enumerate(array):
        priorities.append(value.reshape(int(len(value)**0.5), int(len(value)**0.5)).tolist())
    
    map(lambda x: abs(x - len(graphs[0]-1)), amount)

    for i in amount:
        m = reduce(lambda x,y: x if (x < y) else y, amount)
        if (graphConnectivity(priorities[amount.index(m)])):
            connected.append(amount.index(m))
        else:
            notConnected.append(amount.index(m))
        amount[amount.index(m)] = 999999

    print(connected)
    return connected+notConnected

def generateEvolutionGraphs(parents2, iterations=100, mutationChance=10, fitnessDense = None, fitnessTree = None, fitnessVector = None):
    parents = []

    if (fitnessDense == None and fitnessTree == None and fitnessVector == None):
        print("Fintess function is not set. Random graphs are generated")

    for i in range(len(parents2)):
        parents.append(parents2[i].flatten().tolist())
    # parents = parents.tolist()
    # print(parents)
    # 1) этап - кроссинговер по биту k
    iter = 0
    while iter < iterations:
        iteration = 0

        random.shuffle(parents)
        children = parents.copy()
        while iteration < len(parents)-1:
            k = random.randint(1, len(parents[iteration])-1)
            if (iteration+1<len(parents)-1):
                children[iteration] = parents[iteration][:k] + parents[iteration+1][k:]
                children[iteration+1] = parents[iteration+1][:k] + parents[iteration][k:]
            else:
                children[iteration] = parents[iteration]
            iteration += 2
        # 2) этап - мутация по биту k
        for i in children:
            if (random.randint(1, 100) <= mutationChance):
                k = random.randint(0, len(i)-1)
                i[k] = 0 if (i[k] == 1) else 1


        # 3) этап - отбор
        # fitness функция для плотности графа. 1-плотный; 0-разреженный
        population = parents + children
        parents = []    

        if fitnessDense == None and fitnessTree == None and fitnessVector == None:
            for i in range(int(len(population) / 2)):
                parents.append(population.pop(random.randint(0, len(population)-1)))
        elif fitnessVector != None:
            vectorPriorities = fitnessVectorGraph(population, fitnessVector)
            for i in range((int(len(vectorPriorities) / 2))):
                parents.append(population[vectorPriorities[i]])
        elif fitnessTree == True:
            treePriorities = fitnessTreeGraph(population)
            for i in range((int(len(treePriorities) / 2))):
                parents.append(population[treePriorities[i]])
        elif fitnessDense != None:
            densePriorites = fitnessDenseGraph(population, fitnessDense)
            for i in range((int(len(densePriorites) / 2))):
                parents.append(population[densePriorites[i]])

        iter += 1

    parents = np.array(parents)
    result = []
    for i, value in enumerate(parents):
        result.append(value.reshape(int(len(value)**0.5), int(len(value)**0.5)).tolist())
    return np.array(result)