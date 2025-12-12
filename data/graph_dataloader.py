import torch
import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader

device = torch.device('cuda:0')

def build_graph(node_features, edges):
    batch_size, n_nodes = node_features.shape
    graphs = []

    for i in range(batch_size):
        node, edge = node_features[i], edges[i]

        src, dst = torch.nonzero(edge)  # turn to sparse matrix. threshold=0.5
        weights = edge[src, dst]

        g = dgl.graph((src, dst), num_nodes=n_nodes).to(device)  # generate graph
        g.ndata['feat'], g.edata['w'] = node, weights  # features

        graphs.append(g)

    return dgl.batch(graphs)  # batch를 하나의 큰 그래프로 취급하여 병렬 연산


class GraphDataset(DGLDataset):
    """
        edge.py에서 전처리되어 저장된 .pt 파일(x, y, edge)을 로드하여 제공
    """

    def __init__(self, saved_file_path):
        print(f"Loading {saved_file_path}...")
        data = torch.load(saved_file_path, weights_only=False)
        dataset_name = os.path.splitext(os.path.basename(saved_file_path))[0]

        self.x = torch.tensor(data['x']).float()  # Raw EEG
        self.y = torch.tensor(data['y']).long()
        self.edge = data['edge']  # Adj matrix (PLV)

        super().__init__(name=dataset_name)

    def process(self):
        graphs = []
        n_samples, n_nodes = self.x.shape[0], self.x.shape[1]

        for i in range(n_samples):
            node_features = self.x[i]  # (30, 384)

            adj = self.edge[i]  # (30, 30)

            adj.fill_diagonal_(0)  # self loop 제거

            src, dst = torch.nonzero(adj, as_tuple=True)  # True인 인덱스 추출
            g = dgl.graph((src, dst), num_nodes=n_nodes)

            g.ndata['feat'] = node_features.float()

            weights = adj[src, dst]
            g.edata['w'] = weights

            graphs.append(g)

        self.graphs = graphs
        self.graph_labels = self.y

    def __getitem__(self, item):
        return self.graphs[item], self.graph_labels[item]

    def __len__(self):
        return len(self.graphs)


def get_dataloaders(train_path, test_path, batch_size):
    train_dataset, test_dataset = GraphDataset(train_path), GraphDataset(test_path)
    train_dataloader,  test_dataloader = (GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True),
                                           GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False))

    return train_dataloader, test_dataloader


if __name__ == "__main__":
    import os
    src_path = os.path.join(os.getcwd(), 'data')
    train_path = os.path.join(src_path, 'driver_train_dataset.pt')
    test_path = os.path.join(src_path, 'driver_test_dataset.pt')

    train_dataloader, test_dataloader = get_dataloaders(train_path, test_path, 64)

    for x, y in train_dataloader:
        print(x, y.shape)
