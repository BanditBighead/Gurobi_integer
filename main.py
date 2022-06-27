from gurobipy import *
from util import read_demand_edge
import numpy as np
import time

path_demand = "3_demand.csv"
path_edge = "3_edge.csv"
mode = 0                # 定义优化模式，0为最大化成功业务数，1为最小化所有业务工作路径
cap_edge = 20           # 自定义容量
min_demands = 246       # 在模式1条件下的最小满足需求,容量20最大246，容量10最大156

demand, edge = read_demand_edge(path_demand, path_edge)     # 读出需求和网络
num_node = max(np.max(demand), np.max(edge)) + 1            # 节点数
num_demand = demand.shape[0]                            # 需求数
num_edge = edge.shape[0]                                # 边数
print("num node:", num_node, "  num demand:", num_demand, "  num edge:", num_edge, "  cap edge:", cap_edge)

try:
    # 定义模型
    model = Model('mip')

    count_var = 0       # 参数个数
    # 需求变量列表
    X = []
    for i in range(num_demand):
        X.append(model.addVar(vtype=GRB.BINARY, name=f'x_{i}'))
        count_var += 1

    # 无向边列表(二维列表，第1维表示第i个需求，第2维表示是否占用第j个无向边)
    Y = []
    for i in range(num_demand):
        y_ = []
        for j in range(num_edge):
            y_.append(model.addVar(vtype=GRB.BINARY, name=f'y_{i}_{j}'))
            count_var += 1
        Y.append(y_)

    # 有向边列表(二维列表，第1维表示第i个需求，第2维表示有向边，第j条为正向，第j+num_edge条为同边反向)
    Z = []
    for i in range(num_demand):
        z_ = []
        for j in range(num_edge * 2):
            name = f'z_{i}_{j}'
            z_.append(model.addVar(vtype=GRB.BINARY, name=name))
            count_var += 1
        Z.append(z_)

    print("num param:", count_var)

    # 设置等式约束
    count_equ = 0     # 不等式约束计数
    # 第一个等式
    for i in range(num_demand):
        src = demand[i][0]      # 第i个约束的起点
        dst = demand[i][1]      # 第i个约束的终点
        for k in range(num_node):       # 为每个节点创建约束
            sum_forward = 0     # 当前节点的所有正向和
            sum_backward = 0    # 当前节点的所有反向和
            for j in range(num_edge):
                if edge[j][0] == k:
                    sum_forward += Z[i][j]
                    sum_backward += Z[i][j + num_edge]
                elif edge[j][1] == k:
                    sum_forward += Z[i][j + num_edge]
                    sum_backward += Z[i][j]
            if k == src:
                model.addConstr(sum_forward - sum_backward == X[i], name=f'equ_c{count_equ}')
            elif k == dst:
                model.addConstr(sum_forward - sum_backward == -X[i], name=f'equ_c{count_equ}')
            else:
                model.addConstr(sum_forward - sum_backward == 0, name=f'equ_c{count_equ}')
            count_equ += 1
    # 第二个等式
    for i in range(num_demand):
        for j in range(num_edge):
            model.addConstr(Z[i][j] + Z[i][j + num_edge] == Y[i][j], name=f'equ_c{count_equ}')
            count_equ += 1

    print("constr equ:", count_equ)

    # 设置不等式约束
    count_inequ = 0
    for i in range(num_edge):
        tmp = 0
        for j in range(num_demand):
            tmp += Y[j][i]
        model.addConstr(tmp <= cap_edge, name=f'inequ_c{count_inequ}')
        count_inequ += 1

    if mode == 1:       # 模式1条件下添加满足需求的约束
        model.addConstr(sum(X) >= min_demands, name=f'inequ_c{count_inequ}')
        count_inequ += 1

    print("constr inequ:", count_inequ)

    # 设置优化目标，完成需求数的和，最大化优化
    if mode == 0:
        print("maximize the successful demand")
        model.setObjective(sum(X), GRB.MAXIMIZE)
    else:
        sum_path = 0
        for i in range(num_demand):
            sum_path += sum(Y[i])
        print("minimize the path of successful demand")
        print("the constraint of demands >= ", min_demands)
        model.setObjective(sum_path, GRB.MINIMIZE)

    # 求解
    time_start = time.time()
    model.setParam('outPutFlag', 0)  # 不输出求解日志
    model.optimize()
    time_end = time.time()

    result_X = np.zeros(num_demand)
    result_Y = np.zeros((num_demand, num_edge))
    result_Z = np.zeros((num_demand, num_edge * 2))
    for v in model.getVars():
        if v.varName.startswith('x'):
            num = v.varName.split('_')
            num = int(num[1])
            result_X[num] = v.x
        elif v.varName.startswith('y'):
            num = v.varName.split('_')
            num_x = int(num[1])
            num_y = int(num[2])
            result_Y[num_x][num_y] = v.x
        elif v.varName.startswith('z'):
            num = v.varName.split('_')
            num_x = int(num[1])
            num_y = int(num[2])
            result_Z[num_x][num_y] = v.x

    # 输出
    if mode == 0:
        print('successful demands:', model.objVal, f'  time:{time_end - time_start:.3f} s')
        print('all distance:', result_Y.sum())
    else:
        print('the shortest distance:', model.objVal, f'  time:{time_end - time_start:.3f} s')
        print('successful demand:', result_X.sum())

except GurobiError as e:
    print('Error code '+str(e.errno)+':'+str(e))

except AttributeError:
    print('Encountered an attribute error')


