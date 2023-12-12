import pandas as pd
import numpy as np
import copy

from tqdm import tqdm



def custom_dcgm_distribute(houses, facilities, distance_matrix, selection_range, p=1):

    x, x_id, edge_index, edge_attr = dfs_to_arrays(houses, facilities, distance_matrix)
    flows_distributed = custom_dcgm_loop(edge_index, edge_attr, x, selection_range, p)

    od_matrix = pd.DataFrame(
        np.reshape(flows_distributed, (len(facilities), len(houses))), 
        index=facilities.index, 
        columns=houses.index
        ) 

    print("Flows have been distributed.")
    print("Population left:", x[x[:, 0] == 1, 1].sum())
    print("Capacities left:", x[x[:, 0] == 0, 1].sum())

    return od_matrix


def dfs_to_arrays(houses, facilities, distance_matrix):

    houses = copy.deepcopy(houses)
    facilities = copy.deepcopy(facilities)
    distance_matrix = copy.deepcopy(distance_matrix)

    # transfom dfs into 2d array [[type, value]..]
    houses["type"] = 1
    facilities["type" ]= 0
    houses = houses[["type", "demand"]].rename(columns={"demand": "value"})
    facilities = facilities[["type", "capacity"]].rename(columns={"capacity": "value"})
    nodes = pd.concat([houses, facilities])
    x_id = nodes.index.to_numpy()
    x = nodes[["type", "value"]].to_numpy()

    # make indexes sequential in df and matrix to be able take slice in x by edge_index
    nodes["seq_index"] = range(len(nodes))
    distance_matrix.columns = nodes["seq_index"][list(distance_matrix.columns)]
    distance_matrix.index = nodes["seq_index"][list(distance_matrix.index)]
    distance_matrix = distance_matrix.reindex(sorted(distance_matrix.columns), axis=1).sort_index()

    # represent the connections between houses and facilities as 2d array [[0,0,0,...,n], [1,2,3,..,m]]
    houses_int_id = list(distance_matrix.index)
    services_int_id = list(distance_matrix.columns)
    edge_index = [[], []]
    for i in houses_int_id:
        edge_index[1].extend([i for j in range(len(services_int_id))])
        edge_index[0].extend(services_int_id)
    edge_index = np.array(edge_index)

    # represent the edge attributes (distance) as 2D array based on edge_index 
    edge_attr = np.concatenate(distance_matrix.to_numpy())

    return x, x_id, edge_index, edge_attr


def custom_dcgm_loop(edge_index, edge_attr, x, selection_range, p):
    
    step = selection_range
    max_distant = np.max(edge_attr)

    iter_distances = [step]
    while iter_distances[-1] < max_distant:
        prev_distance = iter_distances[-1]
        iter_distances.append(prev_distance * 2)

    id_edges = np.array(range(len(edge_index[0])))
    flows = np.zeros_like(edge_index[0])
    n = 0

    pbar = tqdm(total=len(iter_distances))
    while n != len(iter_distances): 

        # on each step we select edges for which the following statements are True:
        # 1. edges distance is less than iter_distances[n],
        # 2. source and target have resources (capacity and demand),
        
        select = (edge_attr <= iter_distances[n]) &\
                 (((x[:, 1][edge_index[0]] != 0) & (x[:, 1][edge_index[1]] != 0)))

        edge_index_step = edge_index[:, select]
        edge_attr_step = edge_attr[select]
        id_edges_step = id_edges[select]

        flows_step = gravity_model(edge_index_step, edge_attr_step, x, p)
        if (flows_step == 0).all(): 
            n +=1
            pbar.update(1)

        demand_limit = np.zeros_like(x[:, 1])
        capacity_limit = np.zeros_like(x[:, 1])
        np.add.at(demand_limit, edge_index_step[0], flows_step)
        np.add.at(capacity_limit, edge_index_step[1], flows_step)
        x[:, 1] = x[:, 1] - capacity_limit - demand_limit
        
        np.add.at(flows, id_edges_step, flows_step)
        
    pbar.close()

    return flows


def generate_flows(x, edge_index, y, ax):

    x_int = x[:, 1].astype(int)
    y_sum = np.zeros_like(x[:, 1])
    edge_index_ax = edge_index[ax]

    np.add.at(y_sum, edge_index_ax, y)
    y_norm = y / (y_sum[edge_index_ax] + 3.4028e-38)

    sample_flows = np.zeros_like(y)
    unique_ind = np.unique(edge_index_ax)
    flows_gen = (np.add.at(
        sample_flows,
        np.random.choice((edge_index_ax == i).nonzero()[0], size=x_int[i], p=y_norm[edge_index_ax == i]),
        1
        ) for i in unique_ind if np.sum(y_norm[edge_index_ax == i]) != 0)
        
    list(flows_gen)
    return sample_flows

def normalize_sampling(x, edge_index, y):
    y_norm_i = generate_flows(x, edge_index, y, 0)
    y_norm_j = generate_flows(x, edge_index, y, 1)
    return np.min([y_norm_i, y_norm_j], axis=0)

def gravity_model(edge_index, edge_attr, x, p):
    distance = edge_attr + 1
    y = 1 / distance ** p
    y_norm = normalize_sampling(x, edge_index, y)
    return y_norm
