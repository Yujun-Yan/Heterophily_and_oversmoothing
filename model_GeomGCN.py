import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F

class GeomGCNSingleChannel(nn.Module):
    def __init__(self, g, in_feats, out_feats, num_divisions, activation, dropout_prob, merge):
        super(GeomGCNSingleChannel, self).__init__()
        self.num_divisions = num_divisions
        self.in_feats_dropout = nn.Dropout(dropout_prob)

        self.linear_for_each_division = nn.ModuleList()
        for i in range(self.num_divisions):
            self.linear_for_each_division.append(nn.Linear(in_feats, out_feats, bias=False))

        for i in range(self.num_divisions):
            nn.init.xavier_uniform_(self.linear_for_each_division[i].weight)

        self.activation = activation
        self.g = g
        self.subgraph_edge_list_of_list = self.get_subgraphs(self.g)
        self.merge = merge
        self.out_feats = out_feats

    def get_subgraphs(self, g):
        subgraph_edge_list = [[] for _ in range(self.num_divisions)]
        u, v, eid = g.all_edges(form='all')
        for i in range(g.number_of_edges()):
            subgraph_edge_list[g.edges[u[i], v[i]].data['subgraph_idx']].append(eid[i])

        return subgraph_edge_list

    def forward(self, feature):
        in_feats_dropout = self.in_feats_dropout(feature)
        self.g.ndata['h'] = in_feats_dropout

        for i in range(self.num_divisions):
            subgraph = self.g.edge_subgraph(self.subgraph_edge_list_of_list[i])
            subgraph.copy_from_parent()
            subgraph.ndata[f'Wh_{i}'] = self.linear_for_each_division[i](subgraph.ndata['h']) * subgraph.ndata['norm']
            subgraph.update_all(message_func=fn.copy_u(f'Wh_{i}', out=f'm_{i}'), reduce_func=fn.sum(msg=f'm_{i}', out=f'h_{i}'))
            subgraph.ndata.pop(f'Wh_{i}')
            subgraph.copy_to_parent()

        self.g.ndata.pop('h')
        
        results_from_subgraph_list = []
        for i in range(self.num_divisions):
            if 'h_{}'.format(i) in self.g.node_attr_schemes():
                results_from_subgraph_list.append(self.g.ndata.pop("h_{}".format(i)))
            else:
                results_from_subgraph_list.append(
                    th.zeros((feature.size(0), self.out_feats), dtype=th.float32, device=feature.device))

        if self.merge == 'cat':
            h_new = th.cat(results_from_subgraph_list, dim=-1)
        else:
            h_new = th.mean(th.stack(results_from_subgraph_list, dim=-1), dim=-1)
        h_new = h_new * self.g.ndata['norm'].to(feature.device)
        h_new = self.activation(h_new)
        return h_new

class GeomGCN(nn.Module):
    def __init__(self, g, in_feats, out_feats, num_divisions, activation, num_heads, dropout_prob, ggcn_merge,
                 channel_merge):
        super(GeomGCN, self).__init__()
        self.attention_heads = nn.ModuleList()
        for _ in range(num_heads):
            self.attention_heads.append(
                GeomGCNSingleChannel(g, in_feats, out_feats, num_divisions, activation, dropout_prob, ggcn_merge))
        self.channel_merge = channel_merge
        self.g = g

    def forward(self, feature):
        all_attention_head_outputs = [head(feature) for head in self.attention_heads]
        if self.channel_merge == 'cat':
            return th.cat(all_attention_head_outputs, dim=1)
        else:
            return th.mean(th.stack(all_attention_head_outputs), dim=0)

class GeomGCNNet(nn.Module):
    def __init__(self, g, nlayers, num_input_features, num_output_classes, num_hidden, num_divisions, num_heads,
                 dropout_rate, ggcn_merge, channel_merge, ggcn_merge_last, channel_merge_last):
        super(GeomGCNNet, self).__init__()
        if ggcn_merge == 'cat':
            merge_multiplier = num_divisions
        else:
            merge_multiplier = 1

        if channel_merge == 'cat':
            channel_merge_multiplier = num_heads
        else:
            channel_merge_multiplier = 1
        
        self.convs = nn.ModuleList()
        self.convs.append(GeomGCN(g, num_input_features, num_hidden, num_divisions, F.relu, num_heads,
                                dropout_rate,
                                ggcn_merge, channel_merge))
        for _ in range(nlayers-2):
            self.convs.append(GeomGCN(g, num_hidden * merge_multiplier * channel_merge_multiplier,
                                num_hidden, num_divisions, F.relu,
                                num_heads, dropout_rate, ggcn_merge, channel_merge))

        self.convs.append(GeomGCN(g, num_hidden * merge_multiplier * channel_merge_multiplier,
                                num_output_classes, num_divisions, lambda x: x,
                                num_heads, dropout_rate, ggcn_merge_last, channel_merge_last))
        self.g = g

    def forward(self, features):
        x = features
        for conv in self.convs:
            x = conv(x)
        return F.log_softmax(x, dim=1)