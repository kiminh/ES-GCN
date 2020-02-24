# -*- coding: utf-8 -*-
"""
model-gcn-dgl.ipynb
Authors ASep Fajar Firmansyah
Codes references from 
https://github.com/WeiDongjunGabriel/ESA 
https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/1_gcn.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pygcn.utils import sparse_mx_to_torch_sparse_tensor, normalize 
from dgl_gcn import Net
import scipy.sparse as spT

class ESGCN_DGL(nn.Module):
    def __init__(self, pred2ix_size, pred_embedding_dim, transE_dim, hidden_size, device, label):
        super(ESGCN_DGL, self).__init__()
        self.pred2ix_size = pred2ix_size
        self.pred_embedding_dim = pred_embedding_dim
        self.transE_dim = transE_dim
        self.input_size = self.transE_dim + self.pred_embedding_dim
        #print('input size', self.input_size)
        self.hidden_size = hidden_size
        print('hidden_size', hidden_size)
        #print('label', len(label))
        self.embedding = nn.Embedding(self.pred2ix_size, self.pred_embedding_dim)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, bidirectional=True)
        self.gcn_dgl = Net()
        self.device = device
        
    def forward(self, input_tensor, G):
        # bi-lstm
        pred_embedded = self.embedding(input_tensor[0])
        obj_embedded = input_tensor[1]
        embedded = torch.cat((pred_embedded, obj_embedded), 2)
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)
        lstm_out = torch.flatten(lstm_out, start_dim=1)
        features = normalize(lstm_out.detach().numpy())
        features = torch.FloatTensor(np.array(features))
        
        #gcn-dgl
        g = G
        self.gcn_dgl.train()
        logits = self.gcn_dgl(g, lstm_out)
        output = F.log_softmax(logits, 1)
        return output

    def _init_hidden(self):
        return (torch.randn(2, 1, self.hidden_size, device=self.device), 
            torch.randn(2, 1, self.hidden_size, device=self.device))

