import torch.nn as nn
from torch_geometric.nn.models import GCN
import torch
import numpy as np
import random
torch.use_deterministic_algorithms(True)
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

class lstm_gcn(torch.nn.Module):

    def __init__(self, input_size, hidden_size, num_layers_gcn):
        super(lstm_gcn, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=False)
        self.gcn = GCN(in_channels=input_size, hidden_channels=input_size, num_layers=num_layers_gcn, normalize=False)


    def forward(self, input_aux, edges):
        #print('input: ', input.size())
        input_aux = input_aux.to(next(self.lstm.parameters()).device)
        h0 = torch.zeros(1, input_aux[0].size()[0], input_aux[0].size()[1]).to(next(self.lstm.parameters()).device)
        c0 = torch.zeros(1, input_aux[0].size()[0], input_aux[0].size()[1]).to(next(self.lstm.parameters()).device)
        edges = edges.to(next(self.lstm.parameters()).device)

        output, (hn, cn) = self.lstm.forward(input_aux, (h0, c0))
        #print('hn: ', torch.squeeze(hn).size())
        output = self.gcn.forward(torch.squeeze(hn), edges)
        #print('output: ', output.size())
        return output

