import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import MessagePassing, APPNP
from torch_geometric.nn.conv.gcn_conv import gcn_norm
# -----------------------------------------------------------------------------GCN-------------------------------------------------------------------------------------------------------------
class GCNConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GCNConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, use_iden=False):
        support = torch.mm(input, self.weight)
        if not use_iden:
            output = torch.spmm(adj, support)
        else:
            output = support
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class GCN(nn.Module):
    def __init__(self, nfeat, nlayers, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConvolution(nfeat, nhid))
        for _ in range(nlayers-2):
            self.convs.append(GCNConvolution(nhid, nhid))
        self.convs.append(GCNConvolution(nhid, nclass))
        self.dropout = dropout

    def forward(self, x, adj, get_feat=False):
        for gc in self.convs[:-1]:
            x = F.relu(gc(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        if get_feat:
            x = self.convs[-1](x, adj, True)
        else:
            x = self.convs[-1](x, adj)
        return F.log_softmax(x, dim=1)

#-------------------------------------------------------------------------------------------GCNII------------------------------------------------------------------------------------

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output

class GCNII(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant):
        super(GCNII, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1)

class GCNIIppi(nn.Module):
    def __init__(self, nfeat, nlayers,nhidden, nclass, dropout, lamda, alpha,variant):
        super(GCNIIppi, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant,residual=True))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.sig(self.fcs[-1](layer_inner))
        return layer_inner

##------------------------------------------------------------------------------pair norm--------------------------------------------------------------------
class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, out_features))
       
        self.in_features = in_features
        self.out_features = out_features
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, input, adj):
        h = torch.mm(input, self.weight)
        output = torch.spmm(adj, h)
        if self.bias is not None:
            return output + self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + "({}->{})".format(
                    self.in_features, self.out_features)

class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=1):
        """
            mode:
              'None' : No normalization 
              'PN'   : Original version
              'PN-SI'  : Scale-Individually version
              'PN-SCS' : Scale-and-Center-Simultaneously version
           
            ('SCS'-mode is not in the paper but we found it works well in practice, 
              especially for GCN and GAT.)
            PairNorm is typically used after each graph convolution operation. 
        """
        assert mode in ['None', 'PN',  'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

        # Scale can be set based on origina data, and also the current feature lengths.
        # We leave the experiments to future. A good pool we used for choosing scale:
        # [0.1, 1, 10, 50, 100]
                
    def forward(self, x):
        if self.mode == 'None':
            return x
        
        col_mean = x.mean(dim=0)      
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt() 
            x = self.scale * x / rownorm_mean

        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual

        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        return x

class DeepGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayer=2, residual=0,
                 norm_mode='None', norm_scale=1, **kwargs):
        super(DeepGCN, self).__init__()
        assert nlayer >= 1 
        self.hidden_layers = nn.ModuleList([
            GraphConv(nfeat if i==0 else nhid, nhid) 
            for i in range(nlayer-1)
        ])
        self.out_layer = GraphConv(nfeat if nlayer==1 else nhid , nclass)

        self.dropout = nn.Dropout(p=dropout)
        self.dropout_rate = dropout
        self.relu = nn.ReLU(True)
        self.norm = PairNorm(norm_mode, norm_scale)
        self.skip = residual

    def forward(self, x, adj):
        x_old = 0
        for i, layer in enumerate(self.hidden_layers):
            x = self.dropout(x)
            x = layer(x, adj)
            x = self.norm(x)
            x = self.relu(x)
            if self.skip>0 and i%self.skip==0:
                x = x + x_old
                x_old = x
            
        x = self.dropout(x)
        x = self.out_layer(x, adj)
        return F.log_softmax(x, dim=1)
#-------------------------------------------------------------------------------------------GPRGNN---------------------------------------------------------------------------------
class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class GPRGNN(torch.nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, dprate_GPRGNN, alpha_GPRGNN, Gamma_GPRGNN, Init_GPRGNN, ppnp_GPRGNN):
        super(GPRGNN, self).__init__()
        self.lin1 = nn.Linear(nfeat, nhidden)
        self.lin2 = nn.Linear(nhidden, nclass)

        if ppnp_GPRGNN == 'PPNP':
            self.prop1 = APPNP(nlayers, alpha_GPRGNN)
        elif ppnp_GPRGNN == 'GPR_prop':
            self.prop1 = GPR_prop(nlayers, alpha_GPRGNN, Init_GPRGNN, Gamma_GPRGNN)

        self.Init = Init_GPRGNN
        self.dprate = dprate_GPRGNN
        self.dropout = dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index):

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
#-------------------------------------------------------------------------------------------GAT------------------------------------------------------------------------------------
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks): 
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        # 
        # These are the rows of the second matrix (Wh_repeated_alternating): 
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN 
        # '----------------------------------------------------' -> N times
        # 
        
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj._indices()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        edge_e = self.dropout(edge_e)
        # edge_e: E
        
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(torch.max(e_rowsum, 1e-9*torch.ones_like(e_rowsum)))
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nlayers, nclass, dropout, alpha, nheads, use_sparse=False):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.att_layers = []
        if use_sparse:
            model_sel = SpGraphAttentionLayer
        else:
            model_sel = GraphAttentionLayer
        attentions = [model_sel(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(attentions):
            self.add_module('attention_0_{}'.format(i), attention)
        self.att_layers.append(attentions)
        for j in range(nlayers-2):
            attentions = [model_sel(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
            for i, attention in enumerate(attentions):
                self.add_module('attention_{}_{}'.format(j+1, i), attention)
            self.att_layers.append(attentions)
        self.out_att = model_sel(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        for attentions in self.att_layers:
            x = F.dropout(x, self.dropout, training=self.training)
            x = torch.cat([att(x, adj) for att in attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
    
#-------------------------------------------------------------------------------------------MLP------------------------------------------------------------------------------------    
   
class MLP(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, use_res):
        super(MLP, self).__init__()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.act_fn = F.elu
        self.convs = nn.ModuleList()
        for _ in range(nlayers-2):
            self.convs.append(nn.Linear(nhidden, nhidden))
        self.dropout = dropout
        self.use_res = use_res
        # self.norm = nn.BatchNorm1d(nhidden)
        # self.norm = nn.LayerNorm(nhidden)
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        if self.use_res:
            previous = layer_inner
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            if (i!=0) & self.use_res:
                previous = layer_inner + previous
                # previous = self.norm(previous)
                layer_inner = con(previous)
            else:
                layer_inner = con(layer_inner)
            layer_inner = self.act_fn(layer_inner)
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1)
 #-------------------------------------------------------------------------------------------GGCN------------------------------------------------------------------------------------
class GGCNlayer_SP(nn.Module):
    def __init__(self, in_features, out_features, use_degree=True, use_sign=True, use_decay=True, scale_init=0.5, deg_intercept_init=0.5):
        super(GGCNlayer_SP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fcn = nn.Linear(in_features, out_features)
        self.use_degree = use_degree
        self.use_sign = use_sign
        if use_degree:
            if use_decay:
                self.deg_coeff = nn.Parameter(torch.tensor([0.5,0.0]))
            else:
                self.deg_coeff = nn.Parameter(torch.tensor([deg_intercept_init,0.0]))
        if use_sign:
            self.coeff = nn.Parameter(0*torch.ones([3]))
            self.adj_remove_diag = None
            if use_decay:
                self.scale = nn.Parameter(2*torch.ones([1]))
            else:
                self.scale = nn.Parameter(scale_init*torch.ones([1]))
        self.sftmax = nn.Softmax(dim=-1)
        self.sftpls = nn.Softplus(beta=1)
    
    def precompute_adj_wo_diag(self, adj):
        adj_i = adj._indices()
        adj_v = adj._values()
        adj_wo_diag_ind = (adj_i[0,:]!=adj_i[1,:])
        self.adj_remove_diag = torch.sparse.FloatTensor(adj_i[:,adj_wo_diag_ind], adj_v[adj_wo_diag_ind], adj.size())
                        
    def non_linear_degree(self, a, b, s):
        i = s._indices()
        v = s._values()
        return torch.sparse.FloatTensor(i, self.sftpls(a*v+b), s.size())
    
    def get_sparse_att(self, adj, Wh):
        i = adj._indices()
        Wh_1 = Wh[i[0,:],:]
        Wh_2 = Wh[i[1,:],:]
        sim_vec = F.cosine_similarity(Wh_1, Wh_2)
        sim_vec_pos = F.relu(sim_vec)
        sim_vec_neg = -F.relu(-sim_vec)
        return torch.sparse.FloatTensor(i, sim_vec_pos, adj.size()), torch.sparse.FloatTensor(i, sim_vec_neg, adj.size())
    
    def forward(self, h, adj, degree_precompute):
        if self.use_degree:
            sc = self.non_linear_degree(self.deg_coeff[0], self.deg_coeff[1], degree_precompute)

        Wh = self.fcn(h)
        if self.use_sign:
            if self.adj_remove_diag is None:
                self.precompute_adj_wo_diag(adj)
        if self.use_sign:
            e_pos, e_neg = self.get_sparse_att(adj, Wh)
            if self.use_degree:
                attention_pos = self.adj_remove_diag*sc*e_pos
                attention_neg = self.adj_remove_diag*sc*e_neg
            else:
                attention_pos = self.adj_remove_diag*e_pos
                attention_neg = self.adj_remove_diag*e_neg
            
            prop_pos = torch.sparse.mm(attention_pos, Wh)
            prop_neg = torch.sparse.mm(attention_neg, Wh)
        
            coeff = self.sftmax(self.coeff)
            scale = self.sftpls(self.scale)
            result = scale*(coeff[0]*prop_pos+coeff[1]*prop_neg+coeff[2]*Wh)

        else:
            if self.use_degree:
                prop = torch.sparse.mm(adj*sc, Wh)
            else:
                prop = torch.sparse.mm(adj, Wh)
            
            result = prop
        return result

class GGCNlayer(nn.Module):
    def __init__(self, in_features, out_features, use_degree=True, use_sign=True, use_decay=True, scale_init=0.5, deg_intercept_init=0.5):
        super(GGCNlayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fcn = nn.Linear(in_features, out_features)
        self.use_degree = use_degree
        self.use_sign = use_sign
        if use_degree:
            if use_decay:
                self.deg_coeff = nn.Parameter(torch.tensor([0.5,0.0]))
            else:
                self.deg_coeff = nn.Parameter(torch.tensor([deg_intercept_init,0.0]))
        if use_sign:
            self.coeff = nn.Parameter(0*torch.ones([3]))
            if use_decay:
                self.scale = nn.Parameter(2*torch.ones([1]))
            else:
                self.scale = nn.Parameter(scale_init*torch.ones([1]))
        self.sftmax = nn.Softmax(dim=-1)
        self.sftpls = nn.Softplus(beta=1)


    
    def forward(self, h, adj, degree_precompute):
        if self.use_degree:
            sc = self.deg_coeff[0]*degree_precompute+self.deg_coeff[1]
            sc = self.sftpls(sc)

        Wh = self.fcn(h)
        if self.use_sign:
            prod = torch.matmul(Wh, torch.transpose(Wh, 0, 1))
            sq = torch.unsqueeze(torch.diag(prod),1)
            scaling = torch.matmul(sq, torch.transpose(sq, 0, 1))
            e = prod/torch.max(torch.sqrt(scaling),1e-9*torch.ones_like(scaling))
            e = e-torch.diag(torch.diag(e))
            if self.use_degree:
                attention = e*adj*sc
            else:
                attention = e*adj
            
            attention_pos = F.relu(attention)
            attention_neg = -F.relu(-attention)
            prop_pos = torch.matmul(attention_pos, Wh)
            prop_neg = torch.matmul(attention_neg, Wh)
        
            coeff = self.sftmax(self.coeff)
            scale = self.sftpls(self.scale)
            result = scale*(coeff[0]*prop_pos+coeff[1]*prop_neg+coeff[2]*Wh)

        else:
            if self.use_degree:
                prop = torch.matmul(adj*sc, Wh)
            else:
                prop = torch.matmul(adj, Wh)
            
            result = prop
                 
        return result

        
        
class GGCN(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, decay_rate, exponent, use_degree=True, use_sign=True, use_decay=True, use_sparse=False, scale_init=0.5, deg_intercept_init=0.5, use_bn=False, use_ln=False):
        super(GGCN, self).__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        if use_sparse:
            model_sel = GGCNlayer_SP
        else:
            model_sel = GGCNlayer
        self.convs.append(model_sel(nfeat, nhidden, use_degree, use_sign, use_decay, scale_init, deg_intercept_init))
        for _ in range(nlayers-2):
            self.convs.append(model_sel(nhidden, nhidden, use_degree, use_sign, use_decay, scale_init, deg_intercept_init))
        self.convs.append(model_sel(nhidden, nclass, use_degree, use_sign, use_decay, scale_init, deg_intercept_init))
        self.fcn = nn.Linear(nfeat, nhidden)
        self.act_fn = F.elu
        self.dropout = dropout
        self.use_decay = use_decay
        if self.use_decay:
            self.decay = decay_rate
            self.exponent = exponent
        self.degree_precompute = None
        self.use_degree = use_degree
        self.use_sparse = use_sparse
        self.use_norm = use_bn or use_ln
        if self.use_norm:
            self.norms = nn.ModuleList()
        if use_bn:
            for _ in range(nlayers-1):
                self.norms.append(nn.BatchNorm1d(nhidden))
        if use_ln:
            for _ in range(nlayers-1):
                self.norms.append(nn.LayerNorm(nhidden))
    
    def precompute_degree_d(self, adj):
        diag_adj = torch.diag(adj)
        diag_adj = torch.unsqueeze(diag_adj, dim=1)
        self.degree_precompute = diag_adj/torch.max(adj, 1e-9*torch.ones_like(adj))-1
    
    def precompute_degree_s(self, adj):
        adj_i = adj._indices()
        adj_v = adj._values()
        adj_diag_ind = (adj_i[0,:]==adj_i[1,:])
        adj_diag = adj_v[adj_diag_ind]
        v_new = torch.zeros_like(adj_v)
        for i in range(adj_i.shape[1]):
            v_new[i] = adj_diag[adj_i[0,i]]/adj_v[i]-1
        self.degree_precompute = torch.sparse.FloatTensor(adj_i, v_new, adj.size())
    
    
    def forward(self, x, adj):
        if self.use_degree:
            if self.degree_precompute is None:
                if self.use_sparse:
                    self.precompute_degree_s(adj)
                else:
                    self.precompute_degree_d(adj)
        x = F.dropout(x, self.dropout, training=self.training)
        layer_previous = self.fcn(x)
        layer_previous = self.act_fn(layer_previous)
        layer_inner = self.convs[0](x, adj, self.degree_precompute)

        for i,con in enumerate(self.convs[1:]):
            if self.use_norm:
                layer_inner = self.norms[i](layer_inner)
            layer_inner = self.act_fn(layer_inner)
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            if i==0:
                layer_previous = layer_inner + layer_previous
            else:
                if self.use_decay:
                    coeff = math.log(self.decay/(i+2)**self.exponent+1)
                else:
                    coeff = 1
                layer_previous = coeff*layer_inner + layer_previous
            layer_inner = con(layer_previous,adj,self.degree_precompute)
        return F.log_softmax(layer_inner, dim=1)
    

    
    

if __name__ == '__main__':
    pass






