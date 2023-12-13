import torch
import torch.nn.functional as F
from mp_deterministic import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_sparse import SparseTensor, fill_diag

class OGNNConv(MessagePassing):
    def __init__(self, in_net, fr_net, op_net, cell_net, tm_norm, params):
        super(OGNNConv, self).__init__('mean')
        self.params = params
        self.in_net = in_net
        self.fr_net = fr_net
        self.op_net = op_net
        self.cell_net = cell_net
        self.tm_norm = tm_norm

    def forward(self, x, edge_index, last_cell_state, y):
        if isinstance(edge_index, SparseTensor):
            edge_index = fill_diag(edge_index, fill_value=0)
            if self.params['add_self_loops']==True:
                edge_index = fill_diag(edge_index, fill_value=1)
        else:
            edge_index, _ = remove_self_loops(edge_index)
            if self.params['add_self_loops']==True:
                edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        m = self.propagate(edge_index, x=x, m=None)
        # tm_signal_raw = F.sigmoid(self.tm_net(torch.cat((x, m), dim=1)))
        in_signal_raw = F.sigmoid(self.in_net(torch.cat((x, m), dim=1)))
        fr_signal_raw = F.sigmoid(self.fr_net(torch.cat((x, m), dim=1)))
        op_signal_raw = F.sigmoid(self.op_net(torch.cat((x, m), dim=1)))
        cell_signal_raw = F.tanh(self.cell_net(torch.cat((x, m), dim=1)))

        in_signal = in_signal_raw.repeat_interleave(repeats=int(self.params['hidden_channel']/self.params['chunk_size']), dim=1)
        fr_signal = fr_signal_raw.repeat_interleave(repeats=int(self.params['hidden_channel']/self.params['chunk_size']), dim=1)
        op_signal = op_signal_raw.repeat_interleave(repeats=int(self.params['hidden_channel']/self.params['chunk_size']), dim=1)
        cell_signal = cell_signal_raw.repeat_interleave(repeats=int(self.params['hidden_channel']/self.params['chunk_size']), dim=1)
        out = fr_signal*last_cell_state+in_signal*cell_signal
        out = F.tanh(out)
        out = op_signal*out
        
        return out, cell_signal
    
    def message(self, x_j):
        return x_j
