# -*- coding: utf-8 -*-
"""
model-gcn

model Architecture: 

ESGCN(
  (embedding): Embedding(212, 100)
  (lstm): LSTM(200, 200, bidirectional=True)
  (gcn): GCN(
    (gc1): GraphConvolution (400 -> 32)
    (gc2): GraphConvolution (32 -> 1)
  )
)
Authors ASep Fajar Firmansyah
Codes reference from https://github.com/WeiDongjunGabriel/ESA 

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pygcn.models import GCN
from pygcn.utils import sparse_mx_to_torch_sparse_tensor, normalize 
import scipy.sparse as spT

class ESGCN(nn.Module):
    def __init__(self, pred2ix_size, pred_embedding_dim, transE_dim, hidden_size, nfeatures, nhidden, nclass, dropout, device):
        super(ESGCN, self).__init__()
        self.pred2ix_size = pred2ix_size
        self.pred_embedding_dim = pred_embedding_dim
        self.transE_dim = transE_dim
        self.input_size = self.transE_dim + self.pred_embedding_dim
        self.hidden_size = hidden_size
        self.nfeat = nfeatures
		self.nhid = nhidden
		self.nclass = nclass
		self.dropout = dropout
		self.embedding = nn.Embedding(self.pred2ix_size, self.pred_embedding_dim)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, bidirectional=True)
        self.gcn = GCN(nfeat=self.nfeatures, nhid=self.nhidden, nclass=self.nclass, dropout=self.dropout)
        self.device = device
        self.initial_hidden = self._init_hidden()
        
    def forward(self, input_tensor, G):
        # bi-lstm
        pred_embedded = self.embedding(input_tensor[0])
        obj_embedded = input_tensor[1]
        embedded = torch.cat((pred_embedded, obj_embedded), 2)
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded, self.initial_hidden)
        #lstm_out = lstm_out.permute(1, 0, 2)
        lstm_out = torch.flatten(lstm_out, start_dim=1)
        #print('lstm_out', lstm_out)
        #lstm_out = lstm_out.view(lstm_out.shape[0], -1)
        
        #pygcn
        adj = nx.adjacency_matrix(G)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        #print('adj', adj)

        features = normalize(lstm_out.detach().numpy() )
        features = torch.FloatTensor(np.array(features))
        logits = self.gcn(features, adj)
        #logp = F.log_softmax(logits, 1)
        return logits


    def _init_hidden(self):
        return (torch.randn(2, 1, self.hidden_size, device=self.device), 
            torch.randn(2, 1, self.hidden_size, device=self.device))

