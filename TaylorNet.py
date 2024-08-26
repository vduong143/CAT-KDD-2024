from typing import List, Tuple
from itertools import combinations
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD, AdamW, Adam
from torch.autograd import Variable
import sympy
import tensorly as tl
tl.set_backend('pytorch')

class Fast_Tucker_Taylor(nn.Module):
    def __init__(self,
                    in_features: int,
                    out_features: int = None,
                    X0=None,
                    order=2,
                    rank=50,
                    initial='Taylor') -> None:
        super().__init__()
        out_features = out_features or in_features
        self.order = order
        self.initial = initial
        if X0 is None:
            self.X0 = 0.0
        else:
            assert type(X0) == float or type(X0) == torch.tensor
            self.X0 = X0

        self.const = nn.Parameter(torch.empty((out_features, )))

        self.Os = [
            nn.Parameter(torch.empty((out_features, rank)))
            for i in range(order)
        ]
        self.Is = [
            # nn.Parameter(torch.empty((((i + 1) * rank), in_features)))
            nn.Parameter(torch.empty(((i+1), rank, in_features)))
            for i in range(order)
        ]
        self.Gs = [
            nn.Parameter(torch.empty((rank, rank**(i + 1))))
            for i in range(order)
        ]

        self.Os = nn.ParameterList(self.Os)
        self.Is = nn.ParameterList(self.Is)
        self.Gs = nn.ParameterList(self.Gs)

        self.reset_parameter()

    def reset_parameter(self):
        ''''
        Taylor Initialization Method. If you would like to use
        Taylor initialization, please set ``taylor_init = True``.
        '''
        if self.initial != 'Taylor':
            nn.init.zeros_(self.const)

            if self.initial == 'Xavier':
                for O in self.Os:
                    nn.init.xavier_uniform_(O)
                for I in self.Is:
                    nn.init.xavier_uniform_(I)
                for G in self.Gs:
                    nn.init.xavier_uniform_(G)

            elif self.initial == 'Kaiming':
                for O in self.Os:
                    nn.init.kaiming_uniform_(O)
                for I in self.Is:
                    nn.init.kaiming_uniform_(I)
                for G in self.Gs:
                    nn.init.kaiming_uniform_(G)
        else:
            nn.init.zeros_(self.const)

            for i, O in enumerate(self.Os):
                if i == 0:
                    nn.init.normal_(O, mean=0, std=(1 / (O.shape[1])) ** 0.5)
                else:
                    nn.init.normal_(O, mean=0, std=(0 / (O.shape[1])) ** 0.5)

            def _get_base(in_features, current_order):
                res = 1
                for i in range(current_order):
                    res = res / (in_features + 2 * i)
                return res

            for i, I in enumerate(self.Is):
                exponent = 1/2**(i+1)
                base = _get_base(I.shape[1], i+1)
                nn.init.normal_(I, mean=0, std=base ** exponent)

            for G in self.Gs:
                nn.init.normal_(G, mean=0, std=(1 / (G.shape[1])) ** 0.5)

    def reconstruct(self):
        '''
        Reconsturct the coefficients of Taylor Poly nomials.
        
        return: a list of parameters for each order of taylor layer,
                the shape of each parameter is (out_features, in_features, ... )
        '''
        self.to(torch.device('cpu'))
        rank_i, in_features = self.Is[0].shape[1:]
        out_features, rank_o = self.Os[0].shape
        order = len(self.Os)
        params = [self.const.detach()]

        for i in range(order):
            I = self.Is[i]  # (i+1, rank_i, in_features)
            O = self.Os[i]  # (out_features, rank)
            G = self.Gs[i]  # (rank, rank**(i+1))

            G = G.reshape([rank_o] + [rank_i for _ in range(i + 1)])

            for j in range(0, i + 1):
                # first I puts its rank in the last dimension,
                # which should be inner multiplied by the last dimension of reshaped G
                G = G.transpose(-1 - j, -1)
                G = G @ I[j]
                G = G.transpose(-1 - j, -1)
            # G: rank * in_features * ...
            param = G.transpose(0, -1) @ O.T
            param = param.transpose(0, -1)
            # out_features * in_features * ...
            params.append(param.detach())
        return params

    def forward(self, X: torch.tensor):
        flag_reshape = False
        # If the input is a vector, then reshape it as ``(1, in_features)``.
        # Here, ``1`` can be seemed as batch size.
        if X.dim() == 1:
            X = X.reshape(1, -1)
            flag_reshape = True
        
        # Get basic args from input and attributes.
        batch_size, in_features = X.shape
        out_features, rank_o = self.Os[0].shape
        rank_i = self.Is[0].shape[1]
        order = len(self.Os)

        # if X0 is a tensor, then its length should be the in_features. 
        # Else, we can just seem it as a number.
        if type(self.X0) == torch.tensor:
            assert X.shape[1] == self.X0.shape[0]
        # Here we use the broadcast mechanism in pytorch.
        X = X - self.X0  # .cuda(X.get_device())

        # ``Y`` is the output of this model. Its shape is ``(batch_size, out_features)``.
        # First add constant term to the output.
        Y = torch.zeros((batch_size, out_features)).to(X.device)
        Y = Y + self.const
        
        for i in range(order):
            # ``i`` means the (i+1)-th term in Taylor polynomial.  
            I = self.Is[i]  # (i+1, rank_i, in_features)
            O = self.Os[i]  # (out_features, rank_o)
            G = self.Gs[i]  # (rank, rank**(i+1))
            Z = I @ X.T  # (i+1, rank_i, batch_size)

            # Here we start to calculate Kronecker product
            # If i!=0, then it is not the first order. Then there are 2 more elements.
            # So we should calculate Kronecker product.
            # The Kronecker product is implemented based on broadcast and Hadamard product.
            # ``kron_res`` here means the result of Kronecker product.
            if i != 0:
                kron_res = Z[0] # (rank_i, batch_size)
                for j in range(1, i + 1):
                    X3 = Z[j].reshape([rank_i] + [1 for _ in range(j)] + [batch_size])
                    kron_res = (kron_res.unsqueeze(0) * X3)
            # if i==0, then it means it's the first term.
            # the result of Kronecker product should just be the Z.
            else:
                kron_res = Z
                
            kron_res = kron_res.reshape((-1, batch_size))
            Y = Y + (O @ (G @ kron_res)).T

        if flag_reshape:
            return Y.reshape(-1)
        else:
            return Y


from concept_net import ConceptNet

class TaylorNet(nn.Module):
    def __init__(self,
                num_inputs,
                num_outputs,
                concept_groups=None,
                input_layer='linear',
                hidden_dims=[64,64,32],
                X0=None,
                order=2,
                rank=8,
                initial='Xavier',
                concept_dropout=0.0,
                batchnorm=True,
                encode_concepts=True,
                output_penalty=0.0):
        
        super(TaylorNet, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.concept_dropout = concept_dropout
        self.batchnorm = batchnorm
        self.output_penalty = output_penalty
        self.concept_groups = concept_groups

        if encode_concepts:
            self.concept_nns = ConceptNet(num_inputs, 
                                        hidden_dims,
                                        input_layer=input_layer,
                                        concept_groups=concept_groups,
                                        activation=nn.LeakyReLU(),
                                        dropout=self.concept_dropout,
                                        batchnorm=self.batchnorm)
        else:
            self.concept_nns = None
    
    
        self.taylor_layers = Fast_Tucker_Taylor(num_inputs, 
                                                num_outputs, 
                                                X0=X0, 
                                                order=order, 
                                                rank=rank, 
                                                initial=initial)
    def output_loss(self, output):
        return self.output_penalty * (torch.pow(output, 2).mean(dim=-1)).mean()
  
    def forward(self, x):
        if self.concept_nns:
            z = self.concept_nns(x).squeeze(-1)
        else:
            z = x
        output = self.taylor_layers(z)
        if self.num_outputs == 1:
            output = output.squeeze(-1)

        return output, z

def reconstruct_taylor(model, lamb: float = None):
    params = model.reconstruct()
    print([p.shape for p in params])
    x_len = params[1].shape[1]
    order = len(params) - 1

    zs = sympy.symbols([f'z_{i+1}' for i in range(x_len)])
    z = sympy.Matrix(zs)

    beta = params[0]
    if lamb is not None:
        beta[abs(beta) <= lamb] = 0
    res = [sympy.Matrix(beta)]
    for j in range(order):
        i = j + 1
        Tensori = params[i]
        if lamb is not None:
            Tensori[abs(Tensori) <= lamb] = 0
        kron_z = sympy.kronecker_product(*[z for _ in range(i)])
        ez = sympy.simplify(tl.unfold(Tensori, 0) @ kron_z)
        for p in sympy.preorder_traversal(ez):
            if isinstance(p, sympy.Float):
                ez = ez.subs(p, round(p, 2))
        res.append(ez) 
        
    expression = res[0]
    for i in range(1, len(res)):
        print(res[i])
        expression = expression + res[i]
    expression = sympy.simplify(expression)
    params = [p.detach().cpu().numpy()[0] for p in params]
    return expression, params, zs

def focal_loss(labels, logits, alpha, gamma): 
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss *= modulator
    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)
    focal_loss /= torch.sum(labels)
    return focal_loss


class WeightedLoss(nn.Module):
    def __init__(self, loss_type, samples_per_cls, beta, gamma):
        super(WeightedLoss, self).__init__()
        self.loss_type = loss_type
        self.beta = beta
        self.gamma = gamma
        self.num_cls = len(samples_per_cls)

        effective_samples = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_samples)
        weights = weights / np.sum(weights) * self.num_cls
        self.weights = torch.tensor(weights).float()

    def forward(self, logits, labels):
        labels_one_hot = F.one_hot(labels, self.num_cls).float().to(logits.device)
        weights = self.weights.unsqueeze(0).to(logits.device)
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.num_cls)

        if self.loss_type == "focal":
            loss = focal_loss(labels_one_hot, logits, weights, self.gamma)
        elif self.loss_type == "sigmoid":
            loss = F.binary_cross_entropy_with_logits(logits, labels_one_hot, weight=weights)
        elif self.loss_type == "softmax":
            pred = logits.softmax(dim=1)
            loss = F.binary_cross_entropy(logits, labels_one_hot, weight=weights)
        return loss