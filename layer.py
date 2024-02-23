import torch
import math
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_sparse import SparseTensor, fill_diag

class ONGNNConv(MessagePassing):
    def __init__(self, params):
        super(ONGNNConv, self).__init__(aggr='mean')
        self.params = params
        self.tm_net = torch.nn.Linear(2*params['hidden_channel'], params['chunk_size'])
        self.tm_norm = torch.nn.LayerNorm(params['hidden_channel'])
        self.query = torch.nn.Linear(2*params['hidden_channel'], params['hidden_channel'])
        self.key = torch.nn.Linear(2*params['hidden_channel'], params['hidden_channel'])
        self.value = torch.nn.Linear(2*params['hidden_channel'], params['hidden_channel'])

    def forward(self, x, edge_index, last_tm_signal):
        if isinstance(edge_index, SparseTensor):
            edge_index = fill_diag(edge_index, fill_value=0)
            if self.params['add_self_loops']==True:
                edge_index = fill_diag(edge_index, fill_value=1)
        else:
            edge_index, _ = remove_self_loops(edge_index)
            if self.params['add_self_loops']==True:
                edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        m = self.propagate(edge_index, x=x, m=None)
        m = self.propagate(edge_index, x=x, m=m)

        if self.params['tm']==True:
            if self.params['simple_gating']==True:
                tm_signal_raw = F.sigmoid(self.tm_net(torch.cat((x, m), dim=1)))    
            else:
                tm_signal_raw = F.softmax(self.tm_net(torch.cat((x, m), dim=1)), dim=-1)
                tm_signal_raw = torch.cumsum(tm_signal_raw, dim=-1)
                if self.params['diff_or']==True:
                    tm_signal_raw = last_tm_signal+(1-last_tm_signal)*tm_signal_raw
            tm_signal = tm_signal_raw.repeat_interleave(repeats=int(self.params['hidden_channel']/self.params['chunk_size']), dim=1)
            out = x*tm_signal + m*(1-tm_signal)
        else:
            out = m
            tm_signal_raw = last_tm_signal

        out = self.tm_norm(out)

        return out, tm_signal_raw
    
    def message(self, x_i, x_j, m_i):
        if m_i is None:
            return x_j
        query = self.query(torch.cat((x_i, m_i), dim=1))
        query = F.softmax(query, dim=-1)

        key = self.key(torch.cat((x_j, x_i), dim=1))
        key = F.softmax(key, dim=-1)

        value = self.value(torch.cat((x_j, m_i), dim=1))
        attention = (query * key).sum(-1) / math.sqrt(self.params['hidden_channel'])
        attention = F.leaky_relu(-attention, negative_slope=0.2)
        attention = F.dropout(attention, p=0.2, training=self.training)
        out = attention.view(-1, 1) * value
        return out
    