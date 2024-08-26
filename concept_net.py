from typing import List, Tuple
from itertools import combinations
import torch
from torch import nn
import torch.nn.functional as F

class ExU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ExU, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.Tensor(in_dim))
        self._init_params()

  
    def _init_params(self):
        self.weight = nn.init.normal_(self.weight, mean=4., std=.5)
        self.bias = nn.init.normal_(self.bias, std=.5)
  
    def forward(self, x):
        out = torch.matmul((x - self.bias), torch.exp(self.weight))
        out = torch.clip(out, 0, 1)
        return out

class ConceptNet(nn.Module):
    def __init__(self, 
                num_mlps, 
                hidden_dims, 
                input_layer='linear', 
                concept_groups=None,
                activation=nn.ReLU(),
                order=1, 
                dropout=0.0, 
                batchnorm=False):
    
        super(ConceptNet, self).__init__()
        assert order > 0
        assert (input_layer == "exu" or input_layer == "linear")

        self.num_mlps = num_mlps
        self.model_depth = len(hidden_dims) + 1
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.use_exu = False
        self.use_concept_groups = False
        self.activation = activation

        if concept_groups:
            assert isinstance(concept_groups, list)
            assert len(concept_groups) == num_mlps
            print('Learning {} high-level concepts ...'.format(len(concept_groups)))
            self.use_concept_groups = True
            input_dims = [len(_input) for _input in concept_groups]
            self.concept_groups = concept_groups
        else:
            input_dims = [order for _ in range(num_mlps)]

        layers = []

        # first layer linear (with concept groups) or ExU
        if input_layer == "exu":
            assert concept_groups == None
            assert order == 1
            self.use_exu = True
            self.input_layer = nn.ModuleList([ExU(input_dims[i], hidden_dims[0]) for i in range(num_mlps)])

        elif input_layer == "linear":
            if self.use_concept_groups:
                self.input_layer = nn.ModuleList([nn.Linear(input_dims[i], hidden_dims[0]) for i in range(num_mlps)])
        else:
            self.input_layer = nn.Conv1d(in_channels=order * num_mlps,
                                        out_channels=hidden_dims[0] * num_mlps,
                                        kernel_size=1,
                                        groups=num_mlps)
    
        if self.batchnorm is True:
            layers.append(nn.BatchNorm1d(hidden_dims[0] * num_mlps))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(self.activation)
    
        # followed by linear layers and ReLU
        input_dim = hidden_dims[0]
        for dim in hidden_dims[1:]:
            layers.append(nn.Conv1d(in_channels=input_dim * num_mlps,
                                    out_channels=dim * num_mlps,
                                    kernel_size=1,
                                    groups=num_mlps))
            if self.batchnorm is True:
                layers.append(nn.BatchNorm1d(dim * num_mlps))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(self.activation)
            input_dim = dim

        # last linear layer
        layers.append(nn.Conv1d(in_channels=input_dim * num_mlps,
                                out_channels=1 * num_mlps,
                                kernel_size=1,
                                groups=num_mlps))

        self.hidden_layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_exu:
            Xs = [self.input_layer[i](x[:, i].unsqueeze(1)) for i in range(self.num_mlps)]
            Xs = torch.cat(Xs, dim=1).unsqueeze(-1)
        elif self.use_concept_groups:
            Xs = [self.input_layer[i](x[:, self.concept_groups[i]]) for i in range(self.num_mlps)]
            Xs = torch.cat(Xs, dim=1).unsqueeze(-1)
        else:
            x = x.unsqueeze(-1)
            Xs = self.input_layer(x)
        # print(Xs.shape)
        z = self.hidden_layers(Xs)

        return z
