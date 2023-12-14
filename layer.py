import torch
import torch.nn.functional as F
import torch.nn as nn
from mp_deterministic import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_sparse import SparseTensor, fill_diag

class OGNNConv(MessagePassing):
    def __init__(self, in_net, fr_net, op_net, tm_norm, params):
        super(OGNNConv, self).__init__('mean')
        self.params = params
        self.in_net = in_net
        self.fr_net = fr_net
        self.op_net = op_net
        self.dropout = nn.Dropout(p=params['dropout'])
        self.lin = nn.Linear(params['hidden_channel'], params['hidden_channel'])
        self.att = nn.Parameter(torch.Tensor(3*params['hidden_channel'], 1))
        self.tm_norm = tm_norm

    def forward(self, x, edge_index, last_fr_signal, y):
        if isinstance(edge_index, SparseTensor):
            edge_index = fill_diag(edge_index, fill_value=0)
            if self.params['add_self_loops']==True:
                edge_index = fill_diag(edge_index, fill_value=1)
        else:
            edge_index, _ = remove_self_loops(edge_index)
            if self.params['add_self_loops']==True:
                edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # tm_signal_raw = F.sigmoid(self.tm_net(torch.cat((x, m), dim=1)))
        x = self.lin(x)
        m = self.propagate(edge_index, x=x, size=None, m=None)
        m = self.propagate(edge_index, x=x, size=None, m=m)

        # Cummax
        in_signal_raw = F.softmax(self.in_net(torch.cat((x, m), dim=1)), dim=-1)
        in_signal_raw = torch.cumsum(in_signal_raw, dim=-1)
        fr_signal_raw = F.softmax(self.fr_net(torch.cat((x, m), dim=1)), dim=-1)
        fr_signal_raw = torch.cumsum(fr_signal_raw, dim=-1)

        # Softor
        in_signal_raw = last_fr_signal + (1-last_fr_signal)*in_signal_raw
        fr_signal_raw = in_signal_raw + (1-in_signal_raw)*fr_signal_raw


        in_signal = in_signal_raw.repeat_interleave(repeats=int(self.params['hidden_channel']/self.params['chunk_size']), dim=1)
        fr_signal = fr_signal_raw.repeat_interleave(repeats=int(self.params['hidden_channel']/self.params['chunk_size']), dim=1)

        out = self.op_net(torch.cat((in_signal*x + fr_signal*m, y), dim=1))
        out = self.tm_norm(out)
        
        return out, fr_signal_raw
    
    def message(self, edge_index_i, x_i, x_j, size_i, m_i):
        if (m_i==None):
            return x_j
        else:
            # Compute attention coefficients
            alpha = (torch.cat([x_i, x_j - x_i, m_i-x_i], dim=-1) * self.att).sum(dim=-1)
            alpha = torch.nn.functional.leaky_relu(alpha, self.negative_slope)

            # Apply dropout to attention coefficients
            alpha = torch.nn.functional.dropout(alpha, p=self.dropout, training=self.training)

            # Compute softmax to get attention probabilities
            alpha = torch.nn.functional.softmax(alpha, dim=0)

            # Apply dropout to node features
            x_j = self.dropout_layer(x_j)
            
            # Multiply node features by attention coefficients
            return x_j * alpha.view(-1, self.heads, 1)