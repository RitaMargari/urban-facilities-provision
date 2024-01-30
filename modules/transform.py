import pandas as pd
import geopandas as gpd
import numpy as np
import networkit as nk
import shapely
import geopandas as gpd
import shapely
import copy
import pandas as pd
import shapely.wkt
import networkx as nx
import networkit as nk

from tqdm import tqdm
from scipy import spatial
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

"""Function to convert geometry of nodes / edges from str format to shapely.geometry object"""

def load_graph_geometry(G_nx, node=True, edge=False):

    if edge:
        for u, v, data in G_nx.edges(data=True):
            data["geometry"] = shapely.wkt.loads(data["geometry"])
    if node:
        for u, data in G_nx.nodes(data=True):
            data["geometry"] = shapely.geometry.Point([data["x"], data["y"]])

    return G_nx


"""Functions to calculate distance matrix with road network"""

def calculate_distance_matrix(road_network, houses, facilities, crs=32636, type=['walk'], weight='length_meter'):

    network = road_network.edge_subgraph(
    [(u, v, k) for u, v, k, d in road_network.edges(data=True, keys=True) 
    if d["type"] in type]
    )

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
    net_nk =  convert_nx2nk(network, idmap=nk_idmap, weight=weight)

    # calculate distance matrix
    for i in range(len(splited_matrix)):
        r = nk.distance.SPSP(G=net_nk, sources=splited_matrix[i].index.values).run()
        splited_matrix[i] = splited_matrix[i].apply(lambda x: get_nk_distances(r,x), axis =1)
        del r
        
    distance_matrix = pd.concat(splited_matrix)
    distance_matrix.index = list(facilities.iloc[to_facilities[0][0]].index)
    distance_matrix.columns = list(houses.iloc[from_houses[0][0]].index)
    
    del splited_matrix

    # replace 0 values (caused by road network sparsity) to euclidian distance between two points
    distance_matrix = distance_matrix.progress_apply(lambda x: calculate_euclidian_distance(x, houses, facilities))
    return distance_matrix

def calculate_euclidian_distance(loc, houses, facilities):
    s = copy.deepcopy(loc)
    s_0 = s[s == 0]
    if len(s_0) > 0:
        s.loc[s_0.index] = facilities["geometry"][s.index].distance(houses["geometry"][s.name])
        return s
    else:
        return s
    
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


def get_accessibility_isochrone(mobility_graph, travel_type, x_from, y_from, weight_value, weight_type, crs):
    
    edge_types = {
            "public_transport": ["subway", "bus", "tram", "trolleybus", "walk"],
            "walk": ["walk"], 
            "drive": ["car"]
            }

    travel_names = {
            "public_transport": "Общественный транспорт",
            "walk": "Пешком", 
            "drive": "Личный транспорт"
        }
    
    walk_speed = 4 * 1000 / 60

    mobility_graph = mobility_graph.edge_subgraph(
        [(u, v, k) for u, v, k, d in mobility_graph.edges(data=True, keys=True) 
        if d["type"] in edge_types[travel_type]]
        )
    nodes_data = pd.DataFrame.from_records(
        [d for u, d in mobility_graph.nodes(data=True)], index=list(mobility_graph.nodes())
        ).sort_index()

    distance, start_node = spatial.KDTree(nodes_data[["x", "y"]]).query([x_from, y_from])
    start_node = nodes_data.iloc[start_node].name
    margin_weight = distance / walk_speed if weight_type == "time_min" else distance
    weight_value_remain = weight_value - margin_weight

    weights_sum = nx.single_source_dijkstra_path_length(
        mobility_graph, start_node, cutoff=weight_value_remain, weight=weight_type)
    nodes_data = nodes_data.loc[list(weights_sum.keys())].reset_index()
    nodes_data = gpd.GeoDataFrame(nodes_data, crs=crs)

    if travel_type == "public_transport" and weight_type == "time_min":
        # 0.8 is routes curvature coefficient 
        distance = dict((k, (weight_value_remain - v) * walk_speed * 0.8) for k, v in weights_sum.items())
        nodes_data["left_distance"] = distance.values()
        isochrone_geom = nodes_data["geometry"].buffer(nodes_data["left_distance"])
        isochrone_geom = isochrone_geom.unary_union
    
    else:
        distance = dict((k, (weight_value_remain - v)) for k, v in weights_sum.items())
        nodes_data["left_distance"] = distance.values()
        isochrone_geom = shapely.geometry.MultiPoint(nodes_data["geometry"].tolist()).convex_hull

    isochrone = gpd.GeoDataFrame(
            {"travel_type": [travel_names[travel_type]], "weight_type": [weight_type], 
            "weight_value": [weight_value], "geometry": [isochrone_geom]}).set_crs(crs).to_crs(4326)

    return isochrone


"""Function to transfrom OD and DM matrix to nx graph"""

def create_links(s, OD, dist_limit):
    edges_within = s[s <= dist_limit]
    edges_within_id = list(edges_within.index)
    if len(edges_within_id) > 0:
        edges_within_dist = list(edges_within)
        edges_within_flows = OD[s.name].loc[edges_within_id]
        ebunch = [(s.name, v, {"distance": d, "flows": f}) for v, d, f in zip(
            edges_within_id, edges_within_dist, edges_within_flows
            )]
        return ebunch
    else: 
        return None

def dfs2nx(DM, OD, houses, facilities, dist_limit):
    graph = nx.DiGraph()
    ebunch = DM.progress_apply(lambda s: create_links(s, OD, dist_limit)).dropna().explode().to_list()
    h_nodes = houses.apply(lambda x: {"value": x.demand, "type": 1, "geometry": str(x.geometry)}, axis=1).to_dict()
    f_nodes = facilities.apply(lambda x: {"value": x.capacity, "type": 0, "geometry": str(x.geometry)}, axis=1).to_dict()
    graph.add_edges_from(ebunch)
    nx.set_node_attributes(graph, h_nodes)
    nx.set_node_attributes(graph, f_nodes)
    return graph