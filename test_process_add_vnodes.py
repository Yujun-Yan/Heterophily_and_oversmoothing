import torch
import numpy as np

from process import _compute_neighbourhood_feature_label_distribution, add_vnodes, compute_differences

# Define a simple graph
simple_g = torch.tensor([[0, 1, 1, 0, 0],
                         [1, 0, 0, 1, 1],
                         [1, 0, 0, 0, 1],
                         [0, 1, 0, 0, 1],
                         [0, 1, 1, 1, 0]])

simple_features = torch.tensor([[0, 0],
                                [1, 1],
                                [1, 1],
                                [0, 0],
                                [0, 0]]).to(torch.float)

labels = torch.tensor([0, 1, 1, 0, 0])
# Unit tests for process.py

def test_compute_neighbourhood_feature_label_distribution():
    # Test add_vnodes
    label_neigh_dist, label_feat_mu, label_feat_std = _compute_neighbourhood_feature_label_distribution(simple_g, simple_features, labels)

    # assert something about the result like shapes and values
    # Probability distributions should sum to 1
    print("label_neigh_dist", label_neigh_dist)
    print("label_feat_mu", label_feat_mu)
    print("label_feat_std", label_feat_std)
    assert(torch.allclose(torch.sum(label_neigh_dist[0], dim=0), torch.ones(len(np.unique(labels)))))

def test_add_vnodes():
    num_new_nodes = 2
    new_edges = torch.tensor([[5, 0], [5, 2], [5, 4], [6, 1], [6, 3]])
    new_features = torch.tensor([[2, 2], [3, 3]]).to(torch.float)
    new_labels = torch.tensor([1, 0])

    new_g, new_features, new_labels = add_vnodes(simple_g, simple_features, labels, num_new_nodes, new_edges, new_features, new_labels)
    # print(f'New Graph: {new_g}')
    # print(f'New Features: {new_features}')
    # print(f'New Labels: {new_labels}')

    # assert something about the result like shapes and values
    assert(new_g.shape == (7, 7))
    assert(new_features.shape == (7, 2))
    assert(new_labels.shape == (7,))

def test_diff_from_average_label():
    # Get average label info
    label_neigh_dist, label_feat_mu, label_feat_std = _compute_neighbourhood_feature_label_distribution(simple_g, simple_features, labels)

    # assert something about the result like shapes and values
    # Probability distributions should sum to 1
    print("label_neigh_dist", label_neigh_dist)
    print("label_feat_mu", label_feat_mu)
    print("label_feat_std", label_feat_std)

    # Get differences of all nodes relative to average label
    node_neigh_delta, node_feat_delta = compute_differences(simple_g, simple_features, labels, label_neigh_dist, label_feat_mu, label_feat_std)
    print("node_neigh_delta", node_neigh_delta)
    print("node_feat_delta", node_feat_delta)
    # TODO Assert something...
    assert(node_neigh_delta.shape == (simple_g.shape[0], len(np.unique(labels))))
    assert(node_feat_delta.shape == (simple_g.shape[0], simple_features.shape[1]))

if __name__ == '__main__':
    test_compute_neighbourhood_feature_label_distribution()