import numpy as np
import pulp as pl

m, n = 3, 3

EL = np.array([[3, 4, 2],
               [4, 3, 5],
               [1, 2, 4],
               ])

LA = np.array([3, 4, 2])

second_names = {0:'Ямщиков', 1:'Толмачев', 2:'Горянин'}

prob = pl.LpProblem("", pl.LpMinimize)

x = {(i,j): pl.LpVariable(f"x_{i}_{j}", 0, 1, cat="Binary")
     for i in range(m) for j in range(n)}

C = {(i,j): EL[i,j] / (n * LA[i]) for i in range(m) for j in range(n)}

prob += pl.lpSum(C[i,j] * x[(i,j)] for i in range(m) for j in range(n))

for i in range(m):
    prob += pl.lpSum(x[(i,j)] for j in range(n)) == 1

for j in range(n):
    prob += pl.lpSum(x[(i,j)] for i in range(m)) == 1

prob.solve(pl.PULP_CBC_CMD(msg=False))

X = np.zeros((m,n), dtype=int)
assignments = []

for i in range(m):
    for j in range(n):
        if pl.value(x[(i,j)]) > 0.5:
            X[i][j] = 1
            assignments.append((second_names[i],j+1))

print("Матрица назначений X:\n", X)
print("\nНазначения (исполнитель -> задача):", *assignments, sep='\n')
print("\nОбщие затраты на выполнение задач F:", pl.value(prob.objective))
