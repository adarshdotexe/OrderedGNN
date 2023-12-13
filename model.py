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
        self.in_net = ModuleList()
        self.fr_net = ModuleList()
        self.op_net = ModuleList()

        self.linear_trans_in.append(Linear(params['in_channel'], params['hidden_channel']))


        self.norm_input.append(LayerNorm(params['hidden_channel']))

        for i in range(params['num_layers_input']-1):
            self.linear_trans_in.append(Linear(params['hidden_channel'], params['hidden_channel']))
        self.norm_input.append(LayerNorm(params['hidden_channel']))

        for i in range(params['num_layers']):

            self.tm_norm.append(LayerNorm(params['hidden_channel']))

            if params['simple_gating']==False:
                self.in_net.append(nn.Sequential(
                    nn.Linear(2*params['hidden_channel'], params['chunk_size']),
                    nn.LeakyReLU(),
                    nn.Linear(params['chunk_size'], params['chunk_size']),
                ))
                self.fr_net.append(nn.Sequential(
                    nn.Linear(2*params['hidden_channel'], params['chunk_size']),
                    nn.LeakyReLU(),
                    nn.Linear(params['chunk_size'], params['chunk_size']),
                ))
                self.op_net.append(nn.Sequential(
                    nn.Linear(params['hidden_channel'], params['hidden_channel']),
                    nn.LeakyReLU(),
                    nn.Linear(params['hidden_channel'], params['hidden_channel']),
                ))
            else:
                self.in_net.append(nn.Linear(2*params['hidden_channel'], params['chunk_size']))
                self.fr_net.append(nn.Linear(2*params['hidden_channel'], params['chunk_size']))
                self.op_net.append(nn.Linear(params['hidden_channel'], params['hidden_channel']))
        
            if params['model']=="OGNN":
                self.convs.append(OGNNConv(in_net=self.in_net[i], fr_net=self.fr_net[i], op_net=self.op_net[i], tm_norm=self.tm_norm[i], params=params))

        self.params_conv = list(set(list(self.convs.parameters())+list(self.in_net.parameters())+list(self.fr_net.parameters())+list(self.op_net.parameters())+list(self.tm_norm.parameters())))
        self.params_others = list(self.linear_trans_in.parameters())+list(self.linear_trans_out.parameters())

    def forward(self, x, edge_index):
        check_signal = []

        for i in range(len(self.linear_trans_in)):
            x = F.dropout(x, p=self.params['dropout_rate'], training=self.training)
            x = F.gelu(self.linear_trans_in[i](x))
            x = self.norm_input[i](x)
        y = x
        x+=y

        fr_signal_raw = x.new_zeros(self.params['chunk_size'])

        for j in range(len(self.convs)):
            if self.params['dropout_rate2']!='None':
                x = F.dropout(x, p=self.params['dropout_rate2'], training=self.training)
            else:
                x = F.dropout(x, p=self.params['dropout_rate'], training=self.training)
            x, fr_signal_raw  = self.convs[j](x, edge_index, last_fr_signal=fr_signal_raw, y=y)
            # x+=y
            check_signal.append(dict(zip(['fr_signal'], [fr_signal_raw])))

        x = F.dropout(x, p=self.params['dropout_rate'], training=self.training)
        x = self.linear_trans_out(x)

        encode_values = dict(zip(['x', 'check_signal'], [x, check_signal]))
        
        return encode_values