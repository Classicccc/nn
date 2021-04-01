import random
from functools import reduce
import nntool
import numpy as np
from PIL import Image, ImageFilter

mutationChance = 10
# parents = [[0,1,0,1,0,1,0,0,1], [0,1,1,1,0,0,1,0,0], [1,1,0,1,1,0,1,1,0], [0,0,0,1,1,1,0,1,1]]

parents2 = np.array([[[1, 0, 1, 1, 0, 0, 0, 0, 1, 0,],
 [0, 1, 1, 0, 1, 1, 1, 1, 1, 1,],
 [0, 1, 1, 0, 1, 1, 1, 0, 1, 1,],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 1,],
 [1, 1, 1, 1, 0, 1, 1, 0, 1, 0,],
 [1, 1, 0, 0, 1, 0, 1, 1, 1, 1,],
 [0, 0, 0, 0, 0, 1, 0, 0, 0, 1,],
 [1, 1, 1, 0, 0, 1, 1, 1, 1, 1,],
 [0, 1, 0, 1, 0, 1, 1, 0, 0, 1,],
 [1, 0, 0, 1, 1, 0, 1, 1, 1, 0]],

[[1, 0, 1, 1, 0, 0, 0, 0, 1, 0,],
 [0, 1, 1, 0, 1, 1, 1, 1, 1, 1,],
 [0, 1, 1, 0, 1, 1, 1, 0, 1, 1,],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 1,],
 [1, 1, 1, 1, 0, 1, 1, 0, 1, 0,],
 [1, 1, 0, 0, 1, 0, 1, 1, 1, 1,],
 [0, 0, 0, 0, 0, 1, 0, 0, 0, 1,],
 [1, 1, 1, 0, 0, 1, 1, 1, 1, 1,],
 [0, 1, 0, 1, 0, 1, 1, 0, 0, 1,],
 [1, 0, 0, 1, 1, 0, 1, 1, 1, 0]],

[[1, 0, 1, 1, 0, 0, 0, 0, 1, 0,],
 [0, 1, 1, 0, 1, 1, 1, 1, 1, 1,],
 [0, 1, 1, 0, 1, 1, 1, 0, 1, 1,],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 1,],
 [1, 1, 1, 1, 0, 1, 1, 0, 1, 0,],
 [1, 1, 0, 0, 1, 0, 1, 1, 1, 1,],
 [0, 0, 0, 0, 0, 1, 0, 0, 0, 1,],
 [1, 1, 1, 0, 0, 1, 1, 1, 1, 1,],
 [0, 1, 0, 1, 0, 1, 1, 0, 0, 1,],
 [1, 0, 0, 1, 1, 0, 1, 1, 1, 0]],

 [[1, 1, 1, 1, 0, 0, 0, 0, 1, 0,],
 [0, 1, 1, 0, 1, 1, 1, 1, 1, 1,],
 [0, 1, 1, 0, 1, 1, 1, 0, 1, 1,],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 1,],
 [1, 1, 1, 1, 0, 1, 1, 0, 1, 0,],
 [1, 1, 0, 0, 1, 0, 1, 1, 1, 1,],
 [0, 0, 0, 0, 0, 1, 0, 0, 0, 1,],
 [1, 1, 1, 0, 0, 1, 1, 1, 1, 1,],
 [0, 1, 0, 1, 0, 1, 1, 0, 0, 1,],
 [1, 0, 0, 1, 1, 0, 1, 1, 1, 0]],
 [[1, 0, 1, 1, 0, 0, 0, 0, 1, 0,],
 [0, 1, 1, 0, 1, 1, 1, 1, 1, 1,],
 [0, 1, 1, 0, 1, 1, 1, 0, 1, 1,],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 1,],
 [1, 1, 1, 1, 0, 1, 1, 0, 1, 0,],
 [1, 1, 0, 0, 1, 0, 1, 1, 1, 1,],
 [0, 0, 0, 0, 0, 1, 0, 0, 0, 1,],
 [1, 1, 1, 0, 0, 1, 1, 1, 1, 1,],
 [0, 1, 0, 1, 0, 1, 1, 0, 0, 1,],
 [1, 0, 0, 1, 1, 0, 1, 1, 1, 0]],

[[1, 0, 1, 1, 0, 0, 0, 0, 1, 0,],
 [0, 1, 1, 0, 1, 1, 1, 1, 1, 1,],
 [0, 1, 1, 0, 1, 1, 1, 0, 1, 1,],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 1,],
 [1, 1, 1, 1, 0, 1, 1, 0, 1, 0,],
 [1, 1, 0, 0, 1, 0, 1, 1, 1, 1,],
 [0, 0, 0, 0, 0, 1, 0, 0, 0, 1,],
 [1, 1, 1, 0, 0, 1, 1, 1, 1, 1,],
 [0, 1, 0, 1, 0, 1, 1, 0, 0, 1,],
 [1, 0, 0, 1, 1, 0, 1, 1, 1, 0]],

[[1, 0, 1, 1, 0, 0, 0, 0, 1, 0,],
 [0, 1, 1, 0, 1, 1, 1, 1, 1, 1,],
 [0, 1, 1, 0, 1, 1, 1, 0, 1, 1,],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 1,],
 [1, 1, 1, 1, 0, 1, 1, 0, 1, 0,],
 [1, 1, 0, 0, 1, 0, 1, 1, 1, 1,],
 [0, 0, 0, 0, 0, 1, 0, 0, 0, 1,],
 [1, 1, 1, 0, 0, 1, 1, 1, 1, 1,],
 [0, 1, 0, 1, 0, 1, 1, 0, 0, 1,],
 [1, 0, 0, 1, 1, 0, 1, 1, 1, 0]],

 [[1, 1, 1, 1, 0, 0, 0, 0, 1, 0,],
 [0, 1, 1, 0, 1, 1, 1, 1, 1, 1,],
 [0, 1, 1, 0, 1, 1, 1, 0, 1, 1,],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 1,],
 [1, 1, 1, 1, 0, 1, 1, 0, 1, 0,],
 [1, 1, 0, 0, 1, 0, 1, 1, 1, 1,],
 [0, 0, 0, 0, 0, 1, 0, 0, 0, 1,],
 [1, 1, 1, 0, 0, 1, 1, 1, 1, 1,],
 [0, 1, 0, 1, 0, 1, 1, 0, 0, 1,],
 [1, 0, 0, 1, 1, 0, 1, 1, 1, 0]]])

parents = []

for i in range(len(parents2)):
    parents.append(parents2[i].flatten().tolist())
print(parents)
# parents = parents.tolist()
# print(parents)
# 1) этап - кроссинговер по биту k
iterations = 0
while iterations < 1000:
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
    fitnessResult = []
    parents = []
    for i in population:
        fitnessResult.append(reduce(lambda x,y: x+y, i))
    print(fitnessResult)
    for i in range(int(len(population) / 2)):
        
        m = reduce(lambda x,y: x if (x > y) else y, fitnessResult)
        parents.append(population[fitnessResult.index(m)])
        population.pop(fitnessResult.index(m))
        fitnessResult.pop(fitnessResult.index(m))

    iterations += 1

parents = np.array(parents)
result = []
for i, value in enumerate(parents):
    result.append(value.reshape(int(len(value)**0.5), int(len(value)**0.5)).tolist())
print(np.array(result))
