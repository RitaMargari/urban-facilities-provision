import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx

from tqdm import tqdm
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import os
import torch


class BipartiteData(Data):
    def __init__(self, x_s=None, x_t=None, edge_index=None, y=None, edge_attr=None):
        super().__init__()
        self.x_s = x_s
        self.x_t = x_t
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)
            
    @property
    def num_nodes(self):
        return self.x_s.size(0) + self.x_t.size(1)


class ProvisionSparseDataset(InMemoryDataset):
    def __init__(self, root, transform=None):
        super(ProvisionSparseDataset, self).__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        files = []
        for file in os.listdir(self.root):
            if file.endswith(".graphml"):
                files.append(file)
        return files
        
    @property
    def processed_file_names(self):
        return ['provision.dataset']

    def download(self):
        pass
    
    def process(self):

        f = 4
        
        # read files in specified folder
        houses = gpd.read_file(os.path.join(self.root, "houses.geojson")).set_index("internal_id")
        houses = houses[houses["demand"]!=0]

        facilities = gpd.read_file(os.path.join(self.root, "facilities.geojson")).set_index("internal_id")
        facilities = facilities[facilities["capacity"]!=0]

        OD = pd.read_json(os.path.join(self.root, "od_matrix.json"))
        DM = pd.read_json(os.path.join(self.root, "distance_matrix.json"))

        x = np.array(
            list(houses.apply(lambda x: [x.name, 1, x.demand], axis=1)) + \
            list(facilities.apply(lambda x: [x.name, 0, x.capacity], axis=1))
            )
        edges = np.array(OD.apply(lambda u: [
            [u.name, v, f, d] for v, f, d in zip(OD[u != 0].index, u[u != 0], DM[u.name][u != 0].round())
            ]).explode().dropna().to_list())
            
        houses_id = x[:, 0][x[:, 1] == 1]
        services_id = x[:, 0][x[:, 1] == 0]

        # distance distribution in positive edges
        freq, bins = np.histogram(edges[:, -1], bins=100)
        freq = freq / freq.sum() 

        # undersampling, number of null edges is equal to len(edges) * f
        null_edges = []
        for h_id in houses_id:
            connected_services = edges[:, 1][edges[:, 0] == h_id]
            remain_services = [s for s in services_id if s not in connected_services]
            remain_services_dist = DM[h_id][remain_services]
            
            prob = []
            for dist in list(remain_services_dist): 
                for n in range(len(bins)):
                    if dist <= bins[n] or n == (len(bins) - 1):
                        prob.append(freq[n-1])
                        break

            s_ids = np.random.choice(
                a=remain_services, size=len(connected_services) * f, p=prob/sum(prob), replace=False
                )
            
            distances = list(DM[h_id][s_ids])
            null_edges.extend([[h_id, s_id, 0., d] for s_id, d in zip(s_ids, distances)])

        edges = np.concatenate((edges, np.array(null_edges)))

        # create bipartite graph
        x_s_id, x_t_id = [], []
        x_s, x_t = [], []

        for n, t, v in x:
            if t == 1:
                x_s_id.append(n)
                x_s.append([t, v])
            else:
                x_t_id.append(n)
                x_t.append([t, v])

        y = torch.tensor(edges[:, 2], dtype=torch.float32)
        edge_attr = torch.tensor(edges[:, -1], dtype=torch.float32)
        edge_index = torch.tensor([[
            (np.array(x_s_id) == u).nonzero()[0][0], 
            (np.array(x_t_id) == v).nonzero()[0][0]
            ] for u, v in edges[:, :2]], dtype=torch.long).T
        x_s, x_t = torch.tensor(x_s, dtype=torch.float32), torch.tensor(x_t, dtype=torch.float32)

        # create bipartite torch object          
        graph = BipartiteData(x_s=x_s, x_t=x_t, edge_index=edge_index, y=y, edge_attr=edge_attr)
        graph.x_s_id = torch.tensor(x_s_id, dtype=torch.long)
        graph.x_t_id = torch.tensor(x_t_id, dtype=torch.long)
        
        data_list = []
        data_list.append(graph)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class ProvisionSparseDataset_v2(InMemoryDataset):
    def __init__(self, root, transform=None):
        super(ProvisionSparseDataset_v2, self).__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        files = []
        for file in os.listdir(self.root):
            if file.endswith(".graphml"):
                files.append(file)
        return files
        
    @property
    def processed_file_names(self):
        return ['provision.dataset']

    def download(self):
        pass
    
    def process(self):
        
        data_list = []
        graph = nx.read_graphml(os.path.join(self.root, self.raw_file_names[0]), node_type=int)
        graph_components = [graph.subgraph(c).copy().to_directed() for c in nx.connected_components(graph.to_undirected())]
        for i, component in tqdm(enumerate(graph_components), total=len(graph_components)):
            
            x = gpd.GeoDataFrame.from_dict(dict(component.nodes(data=True)), orient='index')
            x = x.reset_index()[["index", "type", "value"]].to_numpy()
            edges = nx.to_pandas_edgelist(component).to_numpy()

            # create bipartite graph
            x_s_id, x_t_id = [], []
            x_s, x_t = [], []

            for n, t, v in x:
                if t == 1:
                    x_s_id.append(n)
                    x_s.append([t, v])
                else:
                    x_t_id.append(n)
                    x_t.append([t, v])

            y = torch.tensor(edges[:, -1], dtype=torch.float32)
            edge_attr = torch.tensor(edges[:, 2], dtype=torch.float32)
            edge_index = torch.tensor([[
                (np.array(x_s_id) == u).nonzero()[0][0], 
                (np.array(x_t_id) == v).nonzero()[0][0]
                ] for u, v in edges[:, :2]], dtype=torch.long).T
            x_s, x_t = torch.tensor(x_s, dtype=torch.float32), torch.tensor(x_t, dtype=torch.float32)

            # create bipartite torch object          
            part_graph = BipartiteData(x_s=x_s, x_t=x_t, edge_index=edge_index, y=y, edge_attr=edge_attr)
            part_graph.x_s_id = torch.tensor(x_s_id, dtype=torch.long)
            part_graph.x_t_id = torch.tensor(x_t_id, dtype=torch.long)
            part_graph.component = torch.tensor([i])
            
            data_list.append(part_graph)
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])