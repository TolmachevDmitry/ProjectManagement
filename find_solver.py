import numpy as np
import pulp as pl

m, n = 6, 6

EL = np.array([[5, 5, 4, 4, 3, 2],
               [3, 3, 5, 5, 3, 2],
               [2, 2, 2, 2, 5, 5],
               [1, 4, 5, 3, 2, 4],
               [3, 2, 2, 5, 4, 3],
               [5, 3, 3, 1, 2, 4]])

LA = np.array([3, 2, 4, 1, 5, 3])

prob = pl.LpProblem("Assign_One_Task_to_Each", pl.LpMinimize)

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
            assignments.append((i,j))

print("Матрица назначений X:\n", X)
print("\nНазначения (исполнитель -> задача):", assignments)
print("\nОбщие затраты на выполнение задач F:", pl.value(prob.objective))
