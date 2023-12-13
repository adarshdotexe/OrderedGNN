import torch
import torch.nn.functional as F
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
        self.tm_norm = tm_norm

    def forward(self, x, edge_index, last_fr_signal):
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

        out = self.op_net(in_signal*x + fr_signal*m)
        out = self.tm_norm(out)
        
        return out, in_signal_raw, fr_signal_raw
    
    def message(self, x_j):
        return x_j
