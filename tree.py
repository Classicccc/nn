import numpy as np
import nntool
graph = list([[0, 1, 0, 0],[1, 0, 0, 0],[1, 0, 0, 0], [0, 0, 0, 0]])
# visited = np.zeros(len(graph))

# def dfs(u, visited):
#     visitedVertices = 1
#     visited[u] = True
#     for i, value in enumerate(graph[u]):
#         if value == 1 and visited[i]==0:
#             visitedVertices += dfs(i, visited)
#     return visitedVertices

# connect = False
# for i in range(len(graph)):
#     visited = np.zeros(len(graph))
#     if dfs(i, visited) == len(graph):
#         connect = True
#         break

# print(connect)
print(nntool.graphConnectivity(graph))