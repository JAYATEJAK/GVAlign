import torch
from torch import nn
from copy import deepcopy
import torchvision
import numpy as np
import torch.nn.functional as F
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(int(input_dim * 2), int(input_dim)),
            nn.ReLU(),
            nn.Linear(int(input_dim), output_dim)
            # nn.Linear(int(input_dim ), int(input_dim / 2)),
            # nn.ReLU(),
            # nn.Linear(int(input_dim / 2), int(input_dim / 4)),
            # nn.ReLU(),
            # nn.Linear(int(input_dim / 4), output_dim),
            # nn.Sigmoid()
        )
        
    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        # x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

class LLL_Net(nn.Module):
    """Basic class for implementing networks"""

    def __init__(self, model, remove_existing_head=False):
        head_var = model.head_var
        assert type(head_var) == str
        assert not remove_existing_head or hasattr(model, head_var), \
            "Given model does not have a variable called {}".format(head_var)
        assert not remove_existing_head or type(getattr(model, head_var)) in [nn.Sequential, nn.Linear], \
            "Given model's head {} does is not an instance of nn.Sequential or nn.Linear".format(head_var)
        super(LLL_Net, self).__init__()

        self.model = model
        # print(model)
        # print(c)
        last_layer = getattr(self.model, head_var)

        if remove_existing_head:
            if type(last_layer) == nn.Sequential:
                self.out_size = last_layer[-1].in_features
                # strips off last linear layer of classifier
                del last_layer[-1]
            elif type(last_layer) == nn.Linear:
                self.out_size = last_layer.in_features
                # converts last layer into identity
                # setattr(self.model, head_var, nn.Identity())
                # WARNING: this is for when pytorch version is <1.2
                setattr(self.model, head_var, nn.Sequential())
        else:
            self.out_size = last_layer.out_features
        
        print(self.out_size)
        # self.self_sup_head = torchvision.ops.MLP(in_channels = self.out_size,hidden_channels = [self.out_size/2,self.out_size/2,1])
        # self.self_sup_head = MLP(self.out_size,4)
        # hdim = 64
        # self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)
        # print(self.self_sup_head)
        # print(c)

        self.heads = nn.ModuleList()
        self.task_cls = []
        self.task_offset = []
        self._initialize_weights()

    def add_head(self, num_outputs):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
        corresponding offsets
        """
        self.heads.append(nn.Linear(self.out_size, num_outputs))
        # we re-compute instead of append in case an approach makes changes to the heads
        self.task_cls = torch.tensor([head.out_features for head in self.heads])
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])

    def forward(self, x, return_features=False, stochastic=False, cs_stoc=None, manifold_mixup=None, layer_mix=None, target=None, lamda_norm_list=None):
        """Applies the forward pass

        Simplification to work on multi-head only -- returns all head outputs in a list
        Args:
            x (tensor): input images
            return_features (bool): return the representations before the heads
        """

        if manifold_mixup:
            x, y_a, y_b, lam = self.model(x,manifold_mixup=manifold_mixup, layer_mix=layer_mix, target =target, lamda_norm_list = lamda_norm_list)
            assert (len(self.heads) > 0), "Cannot access any head"
            # if stochastic:
            #     assert (len(self.heads) == len(cs_stoc)), "Scaling should match"
            y = []
            k=0
            for head in self.heads:
                if stochastic:
                    y.append(head(x, stochastic))
                # y.append(head(x, stochastic))
                    k +=1
                else: 
                    y.append(head(x, stochastic))

            if return_features:
                return y, x, y_a, y_b, lam
            else:
                return y, y_a, y_b, lam
        else:
            x = self.model(x) 

        
        
        # x = x.unsqueeze(1)
        # breakpoint()

        # x = self.slf_attn(x, x, x)
        # x = x.squeeze(1)
        # breakpoint()
            assert (len(self.heads) > 0), "Cannot access any head"
            # if stochastic:
            #     assert (len(self.heads) == len(cs_stoc)), "Scaling should match"
            y = []
            k=0
            for head in self.heads:
                if stochastic:
                    y.append(head(x, stochastic))
                # y.append(head(x, stochastic))
                    k +=1
                else: 
                    y.append(head(x, stochastic))

            if return_features:
                return y, x
            else:
                return y
            
    def forward_int_nodes(self, x, return_features=False, stochastic=False, cs_stoc=None, manifold_mixup=None, int_nodes_output=None):
        """Applies the forward pass

        Simplification to work on multi-head only -- returns all head outputs in a list
        Args:
            x (tensor): input images
            return_features (bool): return the representations before the heads
        """

        if int_nodes_output:
            x = self.model(x,manifold_mixup=manifold_mixup) 

            assert (len(self.heads) > 0), "Cannot access any head"
            # if stochastic:
            #     assert (len(self.heads) == len(cs_stoc)), "Scaling should match"
            y = []
            k=0
            for head in self.heads:
                if stochastic:
                    y.append(head(x, stochastic))
                # y.append(head(x, stochastic))
                    k +=1
                else: 
                    y.append(head(x, stochastic))
                    
            # breakpoint()
                    
            y_int = self.model.fc_int(x)

            if return_features:
                return y, x, y_int
            else:
                return y, y_int
        else:
            x = self.model(x,manifold_mixup=manifold_mixup) 

            assert (len(self.heads) > 0), "Cannot access any head"
            # if stochastic:
            #     assert (len(self.heads) == len(cs_stoc)), "Scaling should match"
            y = []
            k=0
            for head in self.heads:
                if stochastic:
                    y.append(head(x, stochastic))
                # y.append(head(x, stochastic))
                    k +=1
                else: 
                    y.append(head(x, stochastic))

            if return_features:
                return y, x
            else:
                return y

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        """Freeze all parameters from the main model, but not the heads"""
        for param in self.model.parameters():
            param.requires_grad = False

    def freeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _initialize_weights(self):
        """Initialize weights using different strategies"""
        # TODO: add different initialization strategies
        pass


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output
