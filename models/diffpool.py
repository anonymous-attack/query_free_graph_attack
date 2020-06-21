
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool

from math import ceil

max_nodes = 150

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, add_loop=False, lin=True):
        super(GNN, self).__init__()

        self.add_loop = add_loop

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        #batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))
        x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask)))

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        num_nodes = ceil(0.25 * max_nodes)
        dim = 3
        self.gnn1_pool = GNN(3, 64, num_nodes, add_loop=True)
        self.gnn1_embed = GNN(3, 64, 64, add_loop=True, lin=False)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(dim * 64, 64, num_nodes)
        self.gnn2_embed = GNN(dim * 64, 64, 64, lin=False)

        self.gnn3_embed = GNN(dim * 64, 64, 64, lin=False)

        self.lin1 = torch.nn.Linear(dim * 64, 64)
        self.lin2 = torch.nn.Linear(64, 6)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2


def train(epoch, model, optimizer, train_loader, train_dataset, device):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        if len(data.adj.shape) > 3:
            data_adj = data.adj[:, :, :, 0] + data.adj[:, :, :, 1] + data.adj[:, :, :, 2] + data.adj[:, :, :, 3]
        else:
            data_adj = data.adj
        if data.x == None:
            bastchsize, n_nodes = data.adj.shape[0], data.adj.shape[1]
            data_x = torch.eye(n_nodes).reshape((1, n_nodes, n_nodes)).repeat(bastchsize, 1, 1).to(device)
        else:
            data_x = data.x
        output, _, _ = model(data_x, data_adj, data.mask)
        loss = F.nll_loss(output, data.y.view(-1))
        loss.backward()
        loss_all += data.y.size(0) * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)

@torch.no_grad()
def test(loader, model, device):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)

        if len(data.adj.shape) > 3:
            data_adj = data.adj[:, :, :, 0] + data.adj[:, :, :, 1] + data.adj[:, :, :, 2] + data.adj[:, :, :, 3]
        else:
            data_adj = data.adj
        if data.x == None:
            bastchsize, n_nodes = data.adj.shape[0], data.adj.shape[1]
            data_x = torch.eye(n_nodes).reshape((1, n_nodes, n_nodes)).repeat(bastchsize, 1, 1).to(device)
        else:
            data_x = data.x
        pred = model(data_x, data_adj, data.mask)[0].max(dim=1)[1]

        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)