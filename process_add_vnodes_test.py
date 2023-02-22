import torch
import numpy as np

from process import _compute_neighbourhood_feature_label_distribution

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


if __name__ == '__main__':
    test_compute_neighbourhood_feature_label_distribution()
