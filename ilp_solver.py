import pandas as pd
import pulp


def declare_varables(loc):
    name = loc.name
    nans = loc.isna()
    index = nans[~nans].index
    var = pd.Series(
        [pulp.LpVariable(name = f"route_{name}_{I}", lowBound=0, cat = "Integer") for I in index], 
        index=index, dtype='object')
    loc[~nans] = var
    return loc

def ilp_recursion(houses, facilities, distance_matrix, od_matrix, selection_range): 

    select = distance_matrix[distance_matrix.iloc[:] <= selection_range]
    select = select.apply(lambda x: 1/(x+1), axis = 1)

    variables = select.apply(lambda x: declare_varables(x), axis = 1)

    # define LP problem
    prob = pulp.LpProblem("problem", pulp.LpMaximize)
    for col in variables.columns:
        t = variables[col].dropna().values
        if len(t) > 0: 
            # constraints 1: distribute no more than in residual demand
            prob +=(pulp.lpSum(t) <= houses['demand_left'][col], f"sum_of_capacities_{col}")
        else: pass

    for index in variables.index:
        t = variables.loc[index].dropna().values
        if len(t) > 0:
            # constraints 2: the number of people assigned for facility is no greater than residual capacity
            prob +=(pulp.lpSum(t) <= facilities['capacity_left'][index], f"sum_of_demands_{index}")
        else:pass
    
    # consts are inverse distances
    costs = []
    for index in variables.index:
        t = variables.loc[index].dropna()
        t = t * select.loc[index].dropna()
        costs.extend(t)
    prob +=(pulp.lpSum(costs), "Sum_of_Transporting_Costs" )

    # solve LP problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # parse distributed flows
    to_df = {}
    for var in prob.variables():
        t = var.name.split('_')
        if int(t[1]) in to_df:
            to_df[int(t[1])].update({int(t[2]): var.value()})
        else:
            to_df[int(t[1])] = {int(t[2]): var.value()}

    # accumulate flows   
    result = pd.DataFrame(to_df).transpose()
    left_columns = list(set(set(od_matrix.columns) - set(result.columns)))
    result = result.join(pd.DataFrame(0, columns = left_columns, index = od_matrix.index), how = 'outer')
    od_matrix = od_matrix + result.fillna(0)

    #  substract destributes demand/capacity
    facilities['capacity_left'] = facilities['capacity'].subtract(od_matrix.sum(axis=1) ,fill_value=0)
    houses['demand_left'] = houses['demand'].subtract(od_matrix.sum(axis=0), fill_value = 0)

    # reduce distance matrix
    distance_matrix = distance_matrix.drop(
        index=facilities[facilities['capacity_left'] == 0].index.values,
        columns = houses[houses['demand_left'] == 0].index.values,
        errors = 'ignore'
        )
    
    # increase selection_range (distance/time) 
    selection_range += selection_range

    # execute untill all population is distributed or all facilities are fulled
    if len(distance_matrix.columns) > 0 and len(distance_matrix.index) > 0:
        return ilp_recursion(houses, facilities, distance_matrix, od_matrix, selection_range)
    else: 
        return od_matrix

def ilp_distribute(houses, facilities, distance_matrix, selection_range):
    
    houses["demand_left"] = houses["demand"]
    facilities["capacity_left"] = facilities["capacity"]
    od_matrix = pd.DataFrame(0, index=distance_matrix.index, columns=distance_matrix.columns)
    od_matrix = ilp_recursion(houses, facilities, distance_matrix, od_matrix, selection_range)

    print("Flows have been distributed.")
    print("Population left:", houses['demand_left'].sum())
    print("Capacities left:", facilities['capacity_left'].sum())
    return od_matrix