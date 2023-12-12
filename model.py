from layer import OGNNConv
import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, LayerNorm
import torch.nn as nn

class GONN(Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.linear_trans_in = ModuleList()
        self.linear_trans_out = nn.Sequential(
            nn.Linear(params['hidden_channel'], params['hidden_channel']),
            nn.GELU(),
            nn.Linear(params['hidden_channel'], params['out_channel']),
        )
        self.norm_input = ModuleList()
        self.convs = ModuleList()

        self.tm_norm = ModuleList()
        self.tm_net = ModuleList()
        self.pr_net = ModuleList()

        self.linear_trans_in.append(Linear(params['in_channel'], params['hidden_channel']))


        self.norm_input.append(LayerNorm(params['hidden_channel']))

        for i in range(params['num_layers_input']-1):
            self.linear_trans_in.append(Linear(params['hidden_channel'], params['hidden_channel']))
        self.norm_input.append(LayerNorm(params['hidden_channel']))

        if params['global_gating']==True:
            tm_net = nn.Linear(2*params['hidden_channel'], params['chunk_size'])
            pr_net = nn.Linear(3*params['hidden_channel'], params['hidden_channel'])

        for i in range(params['num_layers']):
            self.tm_norm.append(LayerNorm(params['hidden_channel']))
            
            if params['global_gating']==False:
                self.tm_net.append(nn.Linear(2*params['hidden_channel'], params['chunk_size']))
                self.pr_net.append(nn.Linear(3*params['hidden_channel'], params['hidden_channel']))
            else:
                self.tm_net.append(tm_net)
                self.pr_net.append(pr_net)
        
            if params['model']=="OGNN":
                self.convs.append(OGNNConv(tm_net=self.tm_net[i],pr_net=self.pr_net[i] , tm_norm=self.tm_norm[i], params=params))

        self.params_conv = list(set(list(self.convs.parameters())+list(self.tm_net.parameters())+list(self.pr_net.parameters())))
        self.params_others = list(self.linear_trans_in.parameters())+list(self.linear_trans_out.parameters())

    def forward(self, x, edge_index):
        check_signal = []

        for i in range(len(self.linear_trans_in)):
            x = F.dropout(x, p=self.params['dropout_rate'], training=self.training)
            x = F.gelu(self.linear_trans_in[i](x))
            x = self.norm_input[i](x)
        y = x
        x+=y

        tm_signal = x.new_zeros(self.params['chunk_size'])

        for j in range(len(self.convs)):
            if self.params['dropout_rate2']!='None':
                x = F.dropout(x, p=self.params['dropout_rate2'], training=self.training)
            else:
                x = F.dropout(x, p=self.params['dropout_rate'], training=self.training)
            x, tm_signal = self.convs[j](x, edge_index, last_tm_signal=tm_signal)
            x+=y*(1-tm_signal.repeat_interleave(repeats=int(self.params['hidden_channel']/self.params['chunk_size']), dim=-1))
            check_signal.append(dict(zip(['tm_signal'], [tm_signal])))

        x = F.dropout(x, p=self.params['dropout_rate'], training=self.training)
        x = self.linear_trans_out(x)

        encode_values = dict(zip(['x', 'check_signal'], [x, check_signal]))
        
        return encode_values