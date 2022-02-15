from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from process import *
from utils import *
from model import *
from model_GeomGCN import *
from torch_geometric.data import Data
import dgl

import uuid

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=2, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

parser.add_argument('--dprate_GPRGNN', type=float, default=0.5, help='Dprate for GPRGNN.')

parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='cora', help='dateset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--alpha', type=float, default=0.5, help='alpha_l')

parser.add_argument('--alpha_GPRGNN', type=float, default=0.1, help='alpha for GPRGNN')
parser.add_argument('--Gamma_GPRGNN', default=None, help='Gamma for GPRGNN')

parser.add_argument('--Init_GPRGNN', type=str,
                        choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'],
                        default='PPR', help='Initialization for GPRGNN')
parser.add_argument('--ppnp_GPRGNN', default='GPR_prop',
                        choices=['PPNP', 'GPR_prop'], help='choice of propagation for GPRGNN')

parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
parser.add_argument('--variant', action='store_true', default=False, help='GCNII* model.')
parser.add_argument('--model', type=str, default="GCN", help='choose models: GCN, GCNII, MLP, GAT ,PN, GPRGNN, GGCN(ours)')
parser.add_argument('--alpha_relu', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--row_normalized_adj', action='store_true', default=False, help='choose normalization')
parser.add_argument('--no_degree', action='store_false', default=True, help='do not use degree correction (degree correction only used with symmetric normalization)')
parser.add_argument('--no_sign', action='store_false', default=True, help='do not use signed weights')
parser.add_argument('--no_decay', action='store_false', default=True, help='do not use decaying in the residual connection')
parser.add_argument('--use_bn', action='store_true', default=False, help='use batch norm when not using decaying')
parser.add_argument('--use_ln', action='store_true', default=False, help='use layer norm when not using decaying')
parser.add_argument('--exponent', type=float, default=3.0, help='exponent in the decay function')
parser.add_argument('--decay_rate', type=float, default=1.0, help='decay_rate in the decay function')
parser.add_argument('--use_res', action='store_true', default=False, help='use residual connection for MLP')
parser.add_argument('--use_sparse', action='store_true', default=False, help='use sparse version of GGNN and GAT for large graphs')
parser.add_argument('--scale_init', type=float, default=0.5, help='initial values of scale (when decaying combination is not used)')
parser.add_argument('--deg_intercept_init', type=float, default=0.5, help='initial values of deg_intercept (when decaying combination is not used)')
parser.add_argument('--get_degree', action='store_true', default=False, help='get acc V.S degree (Only support GCN model)')
parser.add_argument('--n_groups', type=int, default=5, help='Number of degree groups.')
################# GeomGCN parameters#########################################################################
parser.add_argument('--ggcn_merge', type=str, default='cat')
parser.add_argument('--channel_merge', type=str, default='cat')
parser.add_argument('--ggcn_merge_last', type=str, default='mean')
parser.add_argument('--channel_merge_last', type=str, default='mean')
parser.add_argument('--num_divisions', type=int, default=9)
parser.add_argument('--learning_rate_decay_patience', type=int, default=50, help='learning rate decay patience (only for GeomGCN baseline)')
parser.add_argument('--learning_rate_decay_factor', type=float, default=0.8, help='only for GeomGCN baseline')
parser.add_argument('--emb', type=str, default='poincare', help='Embedding methods used for GeomGCN baseline, poincare, struc2vec, MDS')


args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid if torch.cuda.is_available() else "cpu")
current_time = time.strftime("%d_%H_%M_%S", time.localtime(time.time()))
checkpt_file = 'pretrained/'+"{}_{}_{}".format(args.model, args.data, current_time)+'.pt'
print(cudaid,checkpt_file)

def get_acc_h_dist(output, out_last2, labels, deg_vec, idx_test, raw_adj, n_groups=args.n_groups):
    ####### nonzero degree nodes mapping ############
    nonzero_ids = (deg_vec!=0)
    deg_max = np.max(deg_vec[nonzero_ids])
    deg_min = np.min(deg_vec[nonzero_ids])
    upper = np.log2(deg_max)
    lower = np.log2(deg_min)
    group_end = np.linspace(lower, upper, num=n_groups+1)
    # print(np.power(2, group_end))
    ##### make sure to include the nodes with the max degree
    group_end[-1] += 1
    group_mapping = np.zeros((n_groups,deg_vec.shape[-1]), dtype=np.bool)
    for i, ind_deg in enumerate(deg_vec):
        if ind_deg!=0:
            g_id = np.argmax(group_end>np.log2(ind_deg))
            if g_id!=0:
                g_id -= 1
                group_mapping[g_id, i] = True
    nodes_in_same_class = torch.eq(torch.unsqueeze(labels,1), torch.unsqueeze(labels,0)).double()
    neib_in_same_class = raw_adj.to_dense()*nodes_in_same_class
    preds = out_last2.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct_same_class_neib = torch.matmul(neib_in_same_class, torch.unsqueeze(correct,1))
    h_whole = torch.squeeze(correct_same_class_neib)/torch.max(torch.tensor(deg_vec), torch.ones_like(torch.tensor(deg_vec)))
    h_whole_ori = torch.sum(neib_in_same_class,-1)/torch.max(torch.tensor(deg_vec), torch.ones_like(torch.tensor(deg_vec)))
    acc_deg = np.zeros((n_groups))
    h_deg = np.zeros((n_groups))
    h_deg_ori = np.zeros((n_groups))
    for j in range(n_groups):
        j_group_test = torch.logical_and(torch.BoolTensor(group_mapping[j,:]), idx_test)
        if torch.any(j_group_test):
            acc_deg[j] = accuracy(output[j_group_test], labels[j_group_test].to(device)) 
            h_deg[j] = torch.mean(h_whole[j_group_test])
            h_deg_ori[j] = torch.mean(h_whole_ori[j_group_test])
        else:
            acc_deg[j] = -1
            h_deg[j] = -1
            h_deg_ori[j] = -1         
    return acc_deg, h_deg, h_deg_ori


def train_step(model,optimizer, features, labels, adj, idx_train, use_geom):
    model.train()
    optimizer.zero_grad()
    if use_geom:
        output = model(features)
    else:
        output = model(features,adj)
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train.item(),acc_train.item()


def validate_step(model,features,labels,adj,idx_val, use_geom):
    model.eval()
    with torch.no_grad():
        if use_geom:
            output = model(features)
        else:
            output = model(features,adj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(),acc_val.item()

def test_step(model, features, labels, adj, idx_test, use_geom, deg_vec, raw_adj):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        if use_geom:
            output = model(features)
        else:
            output = model(features,adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        if deg_vec is not None:
            out_last2 = model(features,adj, True)
            acc_deg, h_deg, h_deg_ori = get_acc_h_dist(output, out_last2, labels, deg_vec, idx_test, raw_adj)
        else:
            acc_deg = None 
            h_deg = None
            h_deg_ori = None
        return loss_test.item(), acc_test.item(), [acc_deg, h_deg, h_deg_ori]
    

def train(datastr,splitstr):
    use_geom=(args.model=='GEOMGCN')
    get_degree = (args.get_degree) & (args.model=="GCN")
    adj, features, labels, idx_train, idx_val, idx_test, num_features, num_labels, deg_vec, raw_adj = full_load_data(datastr,splitstr,args.row_normalized_adj, model_type=args.model, embedding_method=args.emb, get_degree=get_degree)
    # print(torch.sum(torch.ones(idx_train.shape[0])[idx_train])/idx_train.shape[0]) ### check the training percentage
    features = features.to(device)
    adj = adj.to(device)
    if args.model=="GCN":
        model = GCN (nfeat=features.shape[1],
                nlayers=args.layer,
                nhid=args.hidden,
                nclass=num_labels,
                dropout=args.dropout).to(device)
    elif args.model=="GCNII":
        model = GCNII(nfeat=features.shape[1],
                    nlayers=args.layer,
                    nhidden=args.hidden,
                    nclass=num_labels,
                    dropout=args.dropout,
                    lamda = args.lamda, 
                    alpha=args.alpha,
                    variant=args.variant).to(device)
    elif args.model=="GAT":
        model = GAT(nfeat=features.shape[1], 
                    nlayers=args.layer,
                    nhid=args.hidden, 
                    nclass=num_labels,
                    dropout=args.dropout,
                    alpha=args.alpha_relu, 
                    nheads=args.nb_heads, use_sparse=args.use_sparse).to(device)
        if not args.use_sparse:
            adj = adj.to_dense()
    elif args.model=="PN":
        model = DeepGCN(nfeat=features.shape[1], nhid=args.hidden, nclass=num_labels, dropout=args.dropout, nlayer=args.layer, norm_mode="PN").to(device)
    elif args.model=="GPRGNN":
        model = GPRGNN(nfeat=features.shape[1], nlayers=args.layer, nhidden=args.hidden, nclass=num_labels, dropout=args.dropout, dprate_GPRGNN=args.dprate_GPRGNN, alpha_GPRGNN=args.alpha_GPRGNN, Gamma_GPRGNN=args.Gamma_GPRGNN, Init_GPRGNN=args.Init_GPRGNN, ppnp_GPRGNN=args.ppnp_GPRGNN).to(device)
    elif args.model=="GGCN":
        use_degree = (args.no_degree) & (not args.row_normalized_adj)
        use_sign = args.no_sign
        use_decay = args.no_decay
        use_bn = (args.use_bn) & (not use_decay)
        use_ln = (args.use_ln) & (not use_decay) & (not use_bn)
        model = GGCN(nfeat=features.shape[1], nlayers=args.layer, nhidden=args.hidden, nclass=num_labels, dropout=args.dropout, decay_rate=args.decay_rate, exponent=args.exponent, use_degree=use_degree, use_sign=use_sign, use_decay=use_decay, use_sparse=args.use_sparse, scale_init=args.scale_init, deg_intercept_init=args.deg_intercept_init, use_bn=use_bn, use_ln=use_ln).to(device)
        if not args.use_sparse:
            adj = adj.to_dense()
    elif args.model=="MLP":
        model = MLP(nfeat=features.shape[1], nlayers=args.layer, nhidden=args.hidden, nclass=num_labels, dropout=args.dropout, use_res=args.use_res).to(device) 
        adj = adj.to_dense()
    elif args.model=="GEOMGCN":
        adj.set_n_initializer(dgl.init.zero_initializer)
        adj.set_e_initializer(dgl.init.zero_initializer)
        model = GeomGCNNet(g=adj, nlayers=args.layer, num_input_features=features.shape[1], num_output_classes=num_labels, num_hidden=args.hidden,
                     num_divisions=args.num_divisions, num_heads=args.nb_heads, dropout_rate=args.dropout, ggcn_merge=args.ggcn_merge, channel_merge=args.channel_merge, ggcn_merge_last=args.ggcn_merge_last, channel_merge_last=args.channel_merge_last).to(device)
    else:
        raise NotImplementedError

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    
    if use_geom:
        learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                          factor=args.learning_rate_decay_factor,
                                                                          patience=args.learning_rate_decay_patience)
    bad_counter = 0
    best = 999999999
    for epoch in range(args.epochs):
        loss_tra,acc_tra = train_step(model,optimizer,features,labels,adj,idx_train, use_geom)
        loss_val,acc_val = validate_step(model,features,labels,adj,idx_val, use_geom)
        if(epoch+1)%1 == 0: 
            print('Epoch:{:04d}'.format(epoch+1),
                'train',
                'loss:{:.3f}'.format(loss_tra),
                'acc:{:.2f}'.format(acc_tra*100),
                '| val',
                'loss:{:.3f}'.format(loss_val),
                'acc:{:.2f}'.format(acc_val*100))
        if loss_val < best:
            best = loss_val
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break
    
    test_res = test_step(model,features,labels,adj,idx_test, use_geom, deg_vec, raw_adj)
    acc = test_res[1]
    acc_deg, h_deg, h_deg_ori = test_res[-1]
    return acc*100, acc_deg, h_deg, h_deg_ori

t_total = time.time()
acc_list = []
acc_deg_mean = np.zeros((args.n_groups))
h_deg_mean = np.zeros((args.n_groups))
h_deg_ori_mean = np.zeros((args.n_groups))
for i in range(10):
    datastr = args.data
    splitstr = 'splits/'+args.data+'_split_0.6_0.2_'+str(i)+'.npz'
    acc, acc_deg, h_deg, h_deg_ori = train(datastr,splitstr)
    acc_list.append(acc)
    if acc_deg is not None:
        acc_nonzero = (acc_deg!=-1)
        acc_deg_mean[acc_nonzero] = (acc_deg_mean[acc_nonzero]*i+acc_deg[acc_nonzero])/(i+1)
        h_nonzero = (h_deg!=-1)
        h_deg_mean[h_nonzero] = (h_deg_mean[h_nonzero]*i+h_deg[h_nonzero])/(i+1)
        h_ori_nonzero = (h_deg_ori!=-1)
        h_deg_ori_mean[h_ori_nonzero] = (h_deg_ori_mean[h_ori_nonzero]*i+h_deg_ori[h_ori_nonzero])/(i+1)
    print(i,": {:.2f}".format(acc_list[-1]))
print("Train cost: {:.4f}s".format(time.time() - t_total))
print("Test acc.:{:.2f}".format(np.mean(acc_list)))
print("Test std.:{:.2f}".format(np.std(acc_list)))
if (args.get_degree) & (args.model=="GCN"):
    with np.printoptions(precision=2, suppress=True):
        print("Acc deg:{}".format(acc_deg_mean*100))
        print("Homophily deg:{}".format(h_deg_mean))
        print("Original homophily deg:{}".format(h_deg_ori_mean))
        
        
