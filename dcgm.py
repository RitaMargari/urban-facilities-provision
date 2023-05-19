import pandas as pd
import numpy as np

from ipfn import ipfn


def dcgm_distribute(houses, facilities, distance_matrix, distance_func='pow', p=1, therehold=1e-6):
    od_matrix = pd.DataFrame(0, index=distance_matrix.index, columns=distance_matrix.columns)

    if distance_func == "exp":
        od_matrix_init =  np.exp(distance_matrix * p) 
    elif distance_func == "pow":
        od_matrix_init =  1 / (distance_matrix + 1) ** p # + 1 to avoid 0 in a denominator
    else:
        raise ValueError("Undefined distance decay function. Choose between 'exp' and 'pow'.")

    houses["demand_left"] = houses["demand"]
    facilities["capacity_left"] = facilities["capacity"]

    res_demand = houses["demand_left"].sum()
    res_capacity = facilities["capacity_left"].sum()
    while (houses["demand_left"].sum() != 0) & (facilities["capacity_left"].sum() != 0):
        od_matrix_distributed = gravity_loop(houses, facilities, od_matrix_init)

        od_matrix = od_matrix + od_matrix_distributed
        houses["demand_left"] = houses["demand_left"].sub(od_matrix_distributed.sum(axis=0), fill_value=0)
        facilities["capacity_left"] = facilities["capacity_left"].sub(od_matrix_distributed.sum(axis=1), fill_value=0)

        print("Residual demand:", houses["demand_left"].sum(), "Residual capacity:", facilities["capacity_left"].sum())
        
        if (res_demand == houses["demand_left"].sum()) or (res_capacity == facilities["capacity_left"].sum()):
            print("The matrix balancing doesn't converge.")
            break
        else:
            res_demand = houses["demand_left"].sum()
            res_capacity = facilities["capacity_left"].sum()

    return od_matrix


def gravity_loop(houses, facilities, od_matrix_init):
    
    houses_id = houses[houses["demand_left"] != 0].index
    facilities_id = facilities[facilities["capacity_left"] != 0].index

    dimensions = [[0], [1]]
    aggregates = [
        facilities["capacity_left"][facilities_id].to_numpy(), 
        houses["demand_left"][houses_id].to_numpy()
        ]
    
    od_matrix_init = od_matrix_init.loc[facilities_id, houses_id].fillna(0)
    ipf_solver = ipfn.ipfn(od_matrix_init.to_numpy(), aggregates, dimensions, convergence_rate=1e-10)
    ipf_solver.iteration()

    od_matrix_init = pd.DataFrame(np.floor(od_matrix_init), index=od_matrix_init.index, columns=od_matrix_init.columns)

    return od_matrix_init