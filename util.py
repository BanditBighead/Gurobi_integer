import pandas
import numpy as np


def read_demand_edge(path_demand, path_edge):
    data_demand_csv = pandas.read_csv(path_demand)
    data_edge_csv = pandas.read_csv(path_edge)

    data_demand = np.array(data_demand_csv)
    data_edge = np.array(data_edge_csv)

    demand = data_demand[:, 1:]
    edge = data_edge[:, 1:3]

    return demand, edge

