#!/usr/bin/env python
# coding: utf-8


import numpy as np
import scipy.sparse as sp
import scipy.linalg as spl
import os.path as osp
from tqdm import tqdm
from math import ceil
from sklearn.preprocessing import normalize

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


parser = ArgumentParser("attack",
                        formatter_class=ArgumentDefaultsHelpFormatter,
                        conflict_handler='resolve')
parser.add_argument("--dataset", default="ENZYMES",
                    help="The dataset to be perturbed on [ENZYMES, PROTEINS]")
parser.add_argument("--pert-rate", default=0.2, type=float,
                    help='Perturbation rate of edges to be flipped.')
parser.add_argument("--threshold", default=1e-5, type=float,
                    help='Restart threshold of eigen-solutions.')
parser.add_argument('--model', type=str, default='diffpool',
                    help='The target model to be attacked on [gin, diffpool].')
parser.add_argument('--epoch', type=int, default=21,
                    help='The number of epochs. Default: 21 for ENZYMES dataset, 3 for PROTEINS dataset.')
args = parser.parse_args()

data_name = args.dataset
prec_flips = args.pert_rate
threshold = args.threshold
epoch = args.epoch
max_nodes = 150

# seed = 1
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#    torch.cuda.manual_seed_all(seed)

class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes <= max_nodes

path = osp.join(osp.dirname('.'), 'data', data_name+'_dense')
dataset = TUDataset(path, name=data_name, transform=T.ToDense(max_nodes),
                    pre_filter=MyFilter())
dataset = dataset.shuffle()
n = (len(dataset) + 9) // 10
test_dataset = dataset[:n]
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]
test_loader = DenseDataLoader(test_dataset, batch_size=20)
val_loader = DenseDataLoader(val_dataset, batch_size=20)
train_loader = DenseDataLoader(train_dataset, batch_size=20)
# for g in tqdm(test_dataset):
#     print(g.adj.shape)

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
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask, self.add_loop)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask, self.add_loop)))
        x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask, self.add_loop)))

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



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        if len(data.adj.shape) > 3:
            data_adj = data.adj[:,:,:,0] + data.adj[:,:,:,1] + data.adj[:,:,:,2] + data.adj[:,:,:,3]
        else:
            data_adj = data.adj
        if data.x == None:
            bastchsize, n_nodes = data.adj.shape[0], data.adj.shape[1]
            data_x = torch.eye(n_nodes).reshape((1,n_nodes,n_nodes)).repeat(bastchsize, 1, 1).to(device)
        else:
            data_x = data.x
        output, _, _ = model(data_x, data_adj, data.mask)
        loss = F.nll_loss(output, data.y.view(-1))
        loss.backward()
        loss_all += data.y.size(0) * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)

@torch.no_grad()
def test(loader):
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

for epoch in range(1,21):
    train_loss = train(epoch)
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print('Epoch: {:03d}, Train Loss: {:.7f}, '
          'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                       train_acc, test_acc))

def eigen_restart_flips(adj_matrix, D_, candidates, n_flips, vals_org, vecs_org, th=1e-5, restart=True):
    N = len(D_)
    best_edges = []
    for t in tqdm(range(n_flips)):

        flip_indicator = 1 - 2 * np.array(adj_matrix[tuple(candidates.T)])[0]
        eigen_scores = np.zeros(len(candidates))
        delta_eigvals = []
        for x in range(len(candidates)):
            i, j = candidates[x]
            delta_vals_est = flip_indicator[x] * (
                    2 * vecs_org[i] * vecs_org[j] - vals_org * (vecs_org[i] ** 2 + vecs_org[j] ** 2))
            delta_eigvals.append(delta_vals_est)
            vals_est = vals_org + delta_vals_est
            loss_ij = (np.sqrt(np.sum(vals_org ** 2)) - np.sqrt(np.sum(vals_est ** 2))) ** 2
            eigen_scores[x] = loss_ij
        struct_scores = - np.expand_dims(eigen_scores, 1)
        best_edge_ix = np.argsort(struct_scores, axis=0)[0]
        best_edge = candidates[best_edge_ix].squeeze()
        best_edges.append(best_edge)

        vals_org += delta_eigvals[best_edge_ix[0]]

        C = np.zeros(adj_matrix.shape)
        C[best_edge[0]] = - adj_matrix[best_edge[0]].toarray()/D_[best_edge[0],best_edge[0]]
        C[best_edge[1]] = - adj_matrix[best_edge[1]].toarray()/D_[best_edge[1],best_edge[1]]
        if adj_matrix[best_edge[0], best_edge[1]] == 0:
            adj_matrix[best_edge[0], best_edge[1]] = adj_matrix[best_edge[1], best_edge[0]] = 1
            D_[best_edge[0], best_edge[0]] += 1
            D_[best_edge[1], best_edge[1]] += 1
        else:
            adj_matrix[best_edge[0], best_edge[1]] = adj_matrix[best_edge[1], best_edge[0]] = 0
            D_[best_edge[0], best_edge[0]] += -1
            D_[best_edge[1], best_edge[1]] += -1
        C[best_edge[0]] += (adj_matrix[best_edge[0]].toarray()/D_[best_edge[0],best_edge[0]])[0]
        C[best_edge[1]] += (adj_matrix[best_edge[1]].toarray()/D_[best_edge[1],best_edge[1]])[0]

        if 0 in vals_org:
            vecs_org_copy = vecs_org.copy()
            zero_indices = np.where(vals_org==0)[0]
            vecs_org = normalize(vecs_org, axis=0, norm='l2')
            vecs_org[best_edge[0]] = np.sign(vals_org) * vecs_org[best_edge[0]] + np.dot(C[best_edge[0]], vecs_org) / abs(vals_org)
            vecs_org[best_edge[1]] = np.sign(vals_org) * vecs_org[best_edge[1]] + np.dot(C[best_edge[1]], vecs_org) / abs(vals_org)
            for zero_index in zero_indices:
                tmp = np.dot(C, vecs_org_copy[:,zero_index])
                if np.linalg.norm(tmp) == 0:
                    vecs_org[:, zero_index] = tmp
                else:
                    vecs_org[:,zero_index] =  tmp / np.linalg.norm(tmp)
        else:
            vecs_org = normalize(vecs_org, axis=0, norm='l2')
            vecs_org[best_edge[0]] = np.sign(vals_org)*vecs_org[best_edge[0]] + np.dot(C[best_edge[0]], vecs_org) / abs(vals_org)
            vecs_org[best_edge[1]] = np.sign(vals_org)*vecs_org[best_edge[1]] + np.dot(C[best_edge[1]], vecs_org) / abs(vals_org)

        scale = [np.sum(vecs_org[:,i] * vecs_org[:,i] * np.diag(D_)) for i in range(N)]
        vecs_org = vecs_org / np.sqrt(scale)

        if restart:
            orth_matrix = np.dot(vecs_org.T * np.diag(D_), vecs_org)
            acc_pert = (np.sum(abs(orth_matrix))-N) / (N*(N-1))
            # print("{}/{}".format(acc_pert, th))
            if acc_pert>=th or np.isnan(acc_pert):
                vals_org, vecs_org = spl.eigh(adj_matrix.toarray(), np.diag(adj_matrix.sum(0).A1))

        candidates = np.concatenate((candidates[:best_edge_ix[0]],
                                     candidates[best_edge_ix[0] + 1:]))

    best_edges = np.array(best_edges)
    return best_edges

def propose_attack(model, g, prec_flips=0.2, threshold=1e-5):
    if len(g.adj.shape) > 2:
        A = g.adj[:, :, 0] + g.adj[:, :, 1] + g.adj[:, :, 2] + g.adj[:, :, 3]
    else:
        A = g.adj
    A = A.numpy()
    _A_obs = sp.csr_matrix(A)

    _N = _A_obs.shape[0]
    _E = int(np.sum(_A_obs) / 2)

    # generalized eigenvalues/eigenvectors
    A_tilde = _A_obs + sp.eye(_A_obs.shape[0])
    vals_org, vecs_org = spl.eigh(A_tilde.toarray(), np.diag(A_tilde.sum(0).A1))

    n_flips = int(prec_flips*_E)

    # Choose candidate set
    n_candidates = 20000
    candidates = np.random.randint(0, _N, [n_candidates * 5, 2])
    candidates = candidates[candidates[:, 0] < candidates[:, 1]]
    candidates = np.array(list(set(map(tuple, candidates))))
    candidates_all = candidates[:n_candidates]

    # Attack
    flips = eigen_restart_flips(A_tilde.copy(), np.diag(A_tilde.sum(0).A1),
                                            candidates_all, n_flips, vals_org, vecs_org,
                                            th=threshold)

    A_tack = _A_obs.copy().tolil()
    try:
        for e in flips:
            if _A_obs[e[0], e[1]] == 0:
                A_tack[e[0], e[1]] = A_tack[e[1], e[0]] = 1
            else:
                A_tack[e[0], e[1]] = A_tack[e[1], e[0]] = 0
    except IndexError:
        e = flips
        if _A_obs[e[0], e[1]] == 0:
            A_tack[e[0], e[1]] = A_tack[e[1], e[0]] = 1
        else:
            A_tack[e[0], e[1]] = A_tack[e[1], e[0]] = 0

    # performace
    adj_tack = torch.FloatTensor(A_tack.toarray()).to(device)

    if g.x == None:
        g_x = torch.eye(_N).to(device)
        pred = model(g_x, adj_tack)[0].max(dim=1)[1]
    else:
        pred = model(g.x.to(device), adj_tack)[0].max(dim=1)[1]
    correct = pred.eq(g.y.to(device)).sum().item()

    return correct

correct_all = 0
for g in tqdm(test_dataset):
    correct = propose_attack(model, g, prec_flips=prec_flips, threshold=threshold)
    correct_all += correct

acc = correct_all / len(test_dataset)
print("Accuracy: {}; drop: {}".format(acc, (test_acc-acc)/test_acc))


