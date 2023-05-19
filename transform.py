import pandas as pd
import geopandas as gpd
import numpy as np
import networkit as nk
import shapely

from shapely.geometry import LineString


"""Functions to convert Networkx graph to Networkit graph"""

def get_nx2nk_idmap(G_nx):
    idmap = dict((id, u) for (id, u) in zip(G_nx.nodes(), range(G_nx.number_of_nodes())))
    return idmap

def convert_nx2nk(G_nx, idmap=None, weight=None):

    if not idmap:
        idmap = get_nx2nk_idmap(G_nx)
    n = max(idmap.values()) + 1
    edges = list(G_nx.edges())

    if weight:
        G_nk = nk.Graph(n, directed=G_nx.is_directed(), weighted=True)
        for u_, v_ in edges:
                u, v = idmap[u_], idmap[v_]
                d = dict(G_nx[u_][v_])
                if len(d) > 1:
                    for d_ in d.values():
                            v__ = G_nk.addNodes(2)
                            u__ = v__ - 1
                            w = round(d_[weight], 1) if weight in d_ else 1
                            G_nk.addEdge(u, v, w)
                            G_nk.addEdge(u_, u__, 0)
                            G_nk.addEdge(v_, v__, 0)
                else:
                    d_ = list(d.values())[0]
                    w = round(d_[weight], 1) if weight in d_ else 1
                    G_nk.addEdge(u, v, w)
    else:
        G_nk = nk.Graph(n, directed=G_nx.is_directed())
        for u_, v_ in edges:
                u, v = idmap[u_], idmap[v_]
                G_nk.addEdge(u, v)

    return G_nk

def get_nk_distances(nk_dists, loc):
    target_nodes = loc.index
    source_node = loc.name
    distances = [nk_dists.getDistance(source_node, node) for node in target_nodes]
    return pd.Series(data = distances, index = target_nodes)

def get_nk_distances(nk_dists, loc):
    target_nodes = loc.index
    source_node = loc.name
    distances = [nk_dists.getDistance(source_node, node) for node in target_nodes]
    return pd.Series(data = distances, index = target_nodes)


"""Functions to calculate distance matrix with road network"""

def calculate_distance_matrix(network, houses, facilities, crs=32636):

    # find nearest points to objects on road network
    gdf = gpd.GeoDataFrame.from_dict(dict(network.nodes(data=True)), orient='index')
    gdf["geometry"] = gdf.apply(lambda row: shapely.geometry.Point(row.x, row.y), axis=1)
    nodes_gdf = gpd.GeoDataFrame(gdf, geometry = gdf['geometry'], crs = crs)
    from_houses = nodes_gdf['geometry'].sindex.nearest(houses['geometry'], return_distance = True, return_all = False) 
    to_facilities = nodes_gdf['geometry'].sindex.nearest(facilities['geometry'], return_distance = True, return_all = False)
    
    distance_matrix = pd.DataFrame(0, index = to_facilities[0][1], columns = from_houses[0][1])
    splited_matrix = np.array_split(distance_matrix.copy(deep = True), int(len(distance_matrix) / 1000) + 1)
    
    # conver nx graph to nk graph in oder to speed up the calculation
    nk_idmap = get_nx2nk_idmap(network)
    net_nk =  convert_nx2nk(network, idmap=nk_idmap, weight="length_meter")

    # calculate distance matrix
    for i in range(len(splited_matrix)):
        r = nk.distance.SPSP(G=net_nk, sources=splited_matrix[i].index.values).run()
        splited_matrix[i] = splited_matrix[i].apply(lambda x: get_nk_distances(r,x), axis =1)
        del r
        
    distance_matrix = pd.concat(splited_matrix)
    distance_matrix.index = list(facilities.iloc[to_facilities[0][0]].index)
    distance_matrix.columns = list(houses.iloc[from_houses[0][0]].index)
    
    del splited_matrix
    return distance_matrix


    
"""Functions to transform OD matrix to links with geometry field"""

def build_edges(loc, facilities, houses, destination_matrix):
    
    its_idx = loc.name
    con_idx = list(loc[loc != 0].index)
    geom = [LineString((houses["geometry"][loc.name], g)) for g in list(facilities["geometry"][con_idx])]
    dist = [d for d in destination_matrix[loc.name][loc != 0]]
    flows = [f for f in list(loc[loc != 0])]
    if len(con_idx) > 0:
        return [gpd.GeoDataFrame({
            "facility_id": con_idx,
            "house_id": [its_idx for i in range(len(con_idx))],
            "distance": dist,
            "flow": flows,
            "geometry": geom
            })]
    else:
        return None

def od_matrix_to_links(od_matrix, destination_matrix, houses, facilities):
    edges = od_matrix.progress_apply(lambda loc: build_edges(loc, facilities, houses, destination_matrix))
    edges = pd.concat(list(edges.iloc[0].dropna()))
    return edges