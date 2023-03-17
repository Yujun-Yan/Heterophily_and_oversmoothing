import os
import re
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.model_selection import ShuffleSplit
from utils import sys_normalized_adjacency, sparse_mx_to_torch_sparse_tensor, row_normalized_adjacency
import pickle as pkl
import sys
import networkx as nx
import numpy as np
import scipy.sparse as sp
from dgl import DGLGraph
import torch_geometric

from torch.nn import functional as F

# adapted from geom-gcn


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

################ from GCNII/Geom-GCN (with bugs fixed) ###############################################################################################################


def full_load_citation(dataset_str):
    """
    (J+S)
    We don't care about this – this is for citation datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
    tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
    if len(test_idx_range_full) != len(test_idx_range):
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position, mark them
        # Follow H2GCN code
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended
        non_valid_samples = set(test_idx_range_full) - set(test_idx_range)
    else:
        non_valid_samples = set()
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    ############### Follow H2GCN and fix the bug ##############
    non_valid_samples = non_valid_samples.union(
        set(list(np.where(labels.sum(1) == 0)[0])))
    return adj, features, labels, non_valid_samples


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    rowsum = (rowsum == 0)*1+rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

####### codes from the original GeomGCN git repo #################


def process_geom(G, dataset_name, embedding_method):
    embedding_file_path = os.path.join(
        'structural_neighborhood', "outf_nodes_space_relation_{}_{}.txt".format(dataset_name, embedding_method))
    space_and_relation_type_to_idx_dict = {}

    with open(embedding_file_path) as embedding_file:
        for line in embedding_file:
            if line.rstrip() == 'node1,node2	space	relation_type':
                continue
            line = re.split(r'[\t,]', line.rstrip())
            assert (len(line) == 4)
            assert (int(line[0]) in G and int(line[1]) in G)
            if (line[2], int(line[3])) not in space_and_relation_type_to_idx_dict:
                space_and_relation_type_to_idx_dict[(line[2], int(line[3]))] = len(
                    space_and_relation_type_to_idx_dict)
            if G.has_edge(int(line[0]), int(line[1])):
                G.remove_edge(int(line[0]), int(line[1]))
            G.add_edge(int(line[0]), int(line[1]), subgraph_idx=space_and_relation_type_to_idx_dict[
                (line[2], int(line[3]))])

    space_and_relation_type_to_idx_dict['self_loop'] = len(
        space_and_relation_type_to_idx_dict)
    for node in sorted(G.nodes()):
        if G.has_edge(node, node):
            G.remove_edge(node, node)
        G.add_edge(
            node, node, subgraph_idx=space_and_relation_type_to_idx_dict['self_loop'])
    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    g = DGLGraph(adj)
    for u, v, feature in G.edges(data='subgraph_idx'):
        g.edges[g.edge_id(u, v)].data['subgraph_idx'] = torch.tensor([feature])

    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    g.ndata['norm'] = norm.unsqueeze(1)
    return g


def full_load_data(dataset_name, splits_file_path=None, use_raw_normalize=False, model_type=None, embedding_method=None, get_degree=False):
    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        adj, features, labels, non_valid_samples = full_load_citation(
            dataset_name)
        labels = np.argmax(labels, axis=-1)
        features = features.todense()
        G = nx.DiGraph(adj)
    else:
        graph_adjacency_list_file_path = os.path.join(
            'new_data', dataset_name, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join(
            'new_data', dataset_name, 'out1_node_feature_label.txt')

        G = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}

        if dataset_name == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(
                        line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    # Fix uint8 to uint16 for the following line
                    feature_blank[np.array(
                        line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(
                        line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(
                        line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        features = np.array(
            [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        labels = np.array(
            [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
    features = preprocess_features(features)
    if get_degree:
        deg_vec = np.array(adj.sum(1))
        deg_vec = deg_vec.flatten()
        raw_adj = sparse_mx_to_torch_sparse_tensor(adj)
    else:
        deg_vec = None
        raw_adj = None
    if model_type == 'GEOMGCN':
        g = process_geom(G, dataset_name, embedding_method)
    else:
        g = adj
        if use_raw_normalize:
            # never actually used, alway use D^(-1/2) A D^(-1/2)
            g = row_normalized_adjacency(g)
        else:
            g = sys_normalized_adjacency(g)
        g = sparse_mx_to_torch_sparse_tensor(g, model_type)

    with np.load(splits_file_path) as splits_file:
        train_mask = splits_file['train_mask']
        val_mask = splits_file['val_mask']
        test_mask = splits_file['test_mask']

    #################### remove the nodes that the label vectors are all zeros, they aren't assigned to any class ############
    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        for n_i in non_valid_samples:
            if train_mask[n_i]:
                train_mask[n_i] = False
            elif test_mask[n_i]:
                test_mask[n_i] = False
            elif val_mask[n_i]:
                val_mask[n_i] = False

    num_features = features.shape[1]
    num_labels = len(np.unique(labels))
    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    # pi_avg = []
    # for i in range(num_labels):
    #     num_i = np.sum(labels==i)
    #     pi = num_i/(len(labels)-num_i)
    #     t = pi/(1+pi)
    #     pi_avg.append(t*num_i/len(labels))
    #     print("{}: {}".format(i, t))
    # print(np.sum(pi_avg))

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    # TODO call correctly
    # naive_introduce_virtual_nodes(g, features, labels, train_mask,
    #                               val_mask, test_mask, num_features, num_labels, deg_vec, raw_adj)

    return g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels, deg_vec, raw_adj

####### Virtual node addition #################


def _compute_neighbourhood_feature_label_distribution(g, features, labels):
    '''
    Compute the distribution of the features and labels of the neighbours of each node in the graph.
    Label distribution is a K x K matrix where each row represents the probability distribution over neighbourhood labels.
    Feature mu is a K x F matrix where each row represents the ___________________ over neighbourhood features.
    Assumes:
        - labels are integers from 0 to K-1, where K is the number of classes.
    '''
    num_nodes = g.shape[0]
    num_labels = len(torch.unique(labels))
    num_features = features.shape[1]
    unique_labels = torch.unique(labels)

    # K x K matrix where each row represents the probability distribution over neighbourhood labels
    label_neigh_dist = torch.zeros((num_labels, num_labels))
    # K x F matrix where each row represents the ___________________ over neighbourhood features
    label_feat_mu = torch.zeros((num_labels, num_features))

    print(f'Computing label dist and feature mu for {num_nodes} nodes')
    for node in range(num_nodes):
        neighbour_mask = g[node, :]
        neighbour_indices = torch.argwhere(neighbour_mask)  # K x 1
        neighbour_indices = neighbour_indices.squeeze(dim=-1)  # collapse to K

        neighbour_labels = labels[neighbour_indices]  # K?
        label_counts = torch.bincount(neighbour_labels, minlength=num_labels)
        label_neigh_dist[labels[node]] += label_counts

        # num_neighbours x F
        neighbour_features = features[neighbour_indices, :]
        # Get the sum feature vector of each neighbour # NOTE This is the naive appraoch to combining neighbour info
        # add a collapesed 1 x F
        label_feat_mu[labels[node]] += torch.sum(neighbour_features, dim=0)

    # Normalize each row (label) by the total occurances of a label – becomes probability distribution
    # https://pytorch.org/docs/stable/generated/torch.where.html
    totals_for_each = torch.sum(label_neigh_dist, dim=1)
    totals_for_each = torch.unsqueeze(totals_for_each, dim=1)
    # label_neigh_dist / totals_for_each
    label_neigh_dist = torch.div(label_neigh_dist, totals_for_each)
    # Compute mean feature vector for each label
    label_feat_mu = torch.div(label_feat_mu, totals_for_each)

    # Compute standard deviation feature vector over all neighbours of nodes with a given label l
    # NOTE: This is done separately to save memory and avoid storing all the feature vectors for ALL labels at once
    label_feat_std = torch.zeros((num_labels, num_features))
    print(f'Processing unique labels:', unique_labels)
    for l in unique_labels:
        print(f'Computing std for label {l}')
        # labels for each node
        label_mask = labels == l
        # list of node ids with label
        label_indices = torch.argwhere(label_mask)
        label_indices = label_indices.squeeze(dim=-1)
        # go through each of the nodes in label_indices neighbours and add their features to a tensor
        # reduced_adj = g[label_indices, :] # len(label_indices) x F
        neigh_features = torch.tensor([])
        for node_id in label_indices:
            neigh_mask = g[node_id, :]
            neigh_indices = torch.argwhere(neigh_mask)
            neigh_indices = neigh_indices.squeeze(dim=-1)
            neigh_features = torch.cat(
                (neigh_features, features[neigh_indices, :]), dim=0)

        # take standard devation over tensor (dim=1)
        label_feat_std[l] = torch.std(neigh_features, dim=0)
    return label_neigh_dist, label_feat_mu, label_feat_std


def _binarize_tensor(g):
    g_dense = g.to_dense()
    return torch.where(g_dense > 0, torch.ones_like(g_dense), torch.zeros_like(g_dense))


def _binarize_sparse_tensor(sparse_tensor):
    # Get the shape of the input sparse tensor
    shape = sparse_tensor.shape

    # Get the indices and values of the nonzero elements in the input sparse tensor
    indices = sparse_tensor.coalesce().indices()
    values = torch.ones(indices.shape[1])

    # Create a new sparse tensor with the same shape as the input sparse tensor, but with ones in the indices
    binary_tensor = torch.sparse.FloatTensor(indices, values, shape)

    return binary_tensor


def add_vnodes(
    g,
    features,
    labels,
    num_new_nodes,
    new_edges,
    new_features,
    new_labels
):
    '''
    Creates a new graph (represented with an adjacency matrix, feature matrix and label vector) that includes
    the new virtual nodes whose features are given as an (N', F) torch.tensor where N' denotes the number of 
    virtual nodes to introduce, whose features are given in `new_features`.

    Args:

    (+) g (torch.tensor): Assumes raw adjacency matrix `g` (N x N) as a torch.tensor.
    (+) features (torch.tensor): is a (N x F) torch.tensor where each row represents the features of a node.
    ...
    (+) num_new_nodes (int): Self-explanatory.
    (+) new_edges (torch.tensor): (N' x 2) tensor where each row represents a directed edge from the source
                                  to the destination node.
    (+) new_features (torch.tensor): (N' x F) tensor representing the features of the new nodes added.
    (+) new_labels (torch.tensor): (N') tensor representing the labels of the new nodes added.

    Returns: 
    
    1. A new adjacency matrix as a torch.tensor (N + N', N + N') containing the additional virtual nodes
       where N' denotes the number of virtual nodes to introduce.
    2. A new feature matrix as a torch.tensor (N + N', F) where F denotes the feature dimension.
    3. A new label vector as a torch.tensor (N + N') 
    '''
    g_prime = F.pad(g, (0, num_new_nodes), mode='constant', value=0) # (N x (N + N'))
    g_prime = torch.cat((g_prime, torch.zeros((num_new_nodes, g_prime.shape[1]))), dim=0) # (N + N' x (N + N'))
    g_prime[new_edges[:, 0], new_edges[:, 1]] = 1 # Add the new edges to the graph

    features_prime = torch.cat((features, new_features), dim=0) # (N + N' x F)
    labels_prime = torch.cat((labels, new_labels), dim=0) # (N + N')

    return g_prime, features_prime, labels_prime

def compute_differences(
    g,
    features,
    labels,
    # num_features,
    # num_labels,
    # degree_vec,
    label_neigh_dist,
    label_feat_mu,
    label_feat_std,
):
    '''
    Args:

    (+) label_neigh_dist (torch.tensor): (K x K) tensor where K denotes the number of unique labels.
    (+) label_feat_mu (torch.tensor): (K x F) tensor where F denotes the feature dimension.
    (+) label_feat_std (torch.tensor): (K x F) tensor where F denotes the feature dimension.
    
    Compute the difference between a nodes neighbourhood and the neighbourhood of nodes with the same label.
    Takes a given graph with adjacency g, features and labels, and the label neighbourhood distribution and label feature distribution
    Returns the label distance and feature distance between each node and the average neighbourhood of nodes for its label.
    '''
    num_nodes = g.shape[0]
    num_features = features.shape[1]
    unique_labels = torch.unique(labels)
    num_labels = len(unique_labels)
    
    # Store all nodes' label distance and feature distance with mean
    node_neigh_delta = torch.zeros((num_nodes, num_labels)) # N x K
    node_feat_delta = torch.zeros((num_nodes, num_features)) # N x F
    print(f'Computing label dist and feature mu for {num_nodes} nodes')
    # Compute the neighbourhood label and feature distribution of each individual node
    for node in range(num_nodes):
        neighbour_mask = g[node, :]
        neighbour_indices = torch.argwhere(neighbour_mask)  # K x 1
        neighbour_indices = neighbour_indices.squeeze(dim=-1)  # collapse to K
        
        neighbour_labels = labels[neighbour_indices]  # ??
        label_counts = torch.bincount(neighbour_labels, minlength=num_labels)
        label_dist = label_counts / torch.sum(label_counts) # equivalent to label_counts / torch.sum(neighbour_mask) 

        neighbour_features = features[neighbour_indices, :] # num_neighbours x F
        neighbour_features = torch.mean(neighbour_features, dim=0) # F
        
        # Compare that with the average neighbourhood label and feature distribution of nodes with the same label
        average_features = label_feat_mu[labels[node]]
        average_label = label_neigh_dist[labels[node]]
        
        # Compute the difference between all of these distributions directly (not via KL divergence)
        node_neigh_delta[node] = average_label - label_dist
        node_feat_delta[node] = average_features - neighbour_features

    return node_neigh_delta, node_feat_delta

def convert_to_torch_distributions(label_neigh_dist, label_feat_mu, label_feat_std):
    # Construct label distribution objects from each of label_neigh_dist
    label_neigh_dist_objs = []
    for l in label_neigh_dist:
        label_neigh_dist_objs.append(torch.distributions.categorical.Categorical(l))

    # Construct feature distribution objects from each pair from label_feat_mu and label_feat_std
    feat_neigh_dist_objs = []  # list of torch distribution objects
    for mu, std in zip(label_feat_mu, label_feat_std):
        feat_neigh_dist_objs.append(torch.distributions.multivariate_normal.MultivariateNormal(mu, std))
    return label_neigh_dist_objs, feat_neigh_dist_objs

def compute_divergences(
    g,
    features,
    labels,
    num_features,
    num_labels,
    degree_vec,
    label_neigh_dist_objs,
    feat_neigh_dist_objs,
):
    '''
    Given a graph G (as a unweighted adjacency matrix) and a node classification task, for each class k of node in the graph
    generate a distribution over the neighbourhood labels of neighbouring nodes and a distribution over the features
    of each of the classes.
    Then for each node, compute its divergence with the distributions over the neighbourhood labels and features.
    '''
    num_nodes = g.shape[0]

    # Introduce the virtual nodes that connect all nodes with similar distributions of neighbour labels

    # NAIVE 1: Create a list of divergences, between each node's neighbour dists and the average label neighbour dist calculated
    degree_vec = torch.tensor(degree_vec)
    neigh_label_divergences = torch.zeros(num_nodes)
    neigh_feat_divergences = torch.zeros(num_nodes)
    for i, l in enumerate(labels):
        neighbour_mask = g[i, :]
        neighbour_indices = torch.argwhere(neighbour_mask)  # K x 1
        neighbour_indices = neighbour_indices.squeeze(dim=-1)  # collapse to K
        neighbour_labels = labels[neighbour_indices]  # K
        label_counts = torch.bincount(neighbour_labels, minlength=num_labels)

        # Label divergence
        neigh_label_divergences[i] = torch.distributions.kl.kl_divergence(
                label_neigh_dist_objs[l],
                torch.distributions.categorical.Categorical(label_counts / degree_vec[i]))

        # Feature divergence
        neighbour_features = features[neighbour_indices, :]
        neigh_feat_divergences[i] = torch.distributions.kl.kl_divergence(
                feat_neigh_dist_objs[l],
                torch.distributions.multivariate_normal.MultivariateNormal(
                    torch.mean(neighbour_features, dim=0),
                    torch.std(neighbour_features, dim=0)))

    # NAIVE 1: sub-approach 1: Add virtual nodes to nodes with divergence > epsilon
    # eps = ...

    # NAIVE 1: sub-approach 2: Add virtual nodes to p proportion of nodes with highest divergence
    p = 0.1
    num_vnodes = int(p * num_nodes)

    kl_sorted_label_indices = torch.argsort(neigh_label_divergences)[:num_vnodes]
    kl_sorted_feat_indices = torch.argsort(neigh_feat_divergences)[:num_vnodes]

    return kl_sorted_label_indices, kl_sorted_feat_indices

def naive_strategy_1(
    g,
    features,
    labels,
    num_features,
    num_labels,
    degree_vec,
):
    '''
    Given a weighted sparse adj graph G, a feature matrix and a label vector, add virtual nodes to the graph.
    Strategy 1: connect nodes with high divergence in their distributions of neighbour labels and features, to virtual nodes of a label
    that would help them move towards the average.
    
    '''
    g = _binarize_tensor(g) # Transform (sparse) weighted adj to (dense) unweighted adj
    
    # Compute the distribution of the features and labels of the neighbours of each node in the graph.
    label_neigh_dist, label_feat_mu, label_feat_std = _compute_neighbourhood_feature_label_distribution(
        g, features, labels)
    label_neigh_dist_objs, feat_neigh_dist_objs = convert_to_torch_distributions(label_neigh_dist, label_feat_mu, label_feat_std)

    # TODO: compute_divergences() or compute_differences() ...?
    node_neigh_delta, node_feat_delta = compute_differences(g, features, labels, label_neigh_dist, label_feat_mu, label_feat_std)

    # TODO: Decide which new nodes we are going to add
    raw_adj = None
    num_new_nodes = 0

    g, features, labels = add_vnodes(g, features, labels, num_new_nodes, num_labels, degree_vec, raw_adj)
    return g, features, labels, num_features, num_labels, degree_vec, raw_adj
