# -*- coding: utf-8 -*-
import numpy as np
import json,pickle,time,os
from scipy.sparse import coo_matrix
import torch
from align_gat import align_cca
from gat import GAT
import torch.nn.functional as F
from utils import dataset,get_sim,hit_precision
from multiprocessing import Pool
from functools import partial
import networkx as nx

def Laplacian_graph(A):
    for i in range(len(A)):
        A[i, i] = 1
    A = torch.FloatTensor(A)
    D_ = torch.diag(torch.sum(A, 0)**(-0.5))
    A_hat = torch.matmul(torch.matmul(D_,A),D_)
    A_hat = A_hat.float()
    indices = torch.nonzero(A_hat).t()
    values = A_hat[indices[0], indices[1]]
    A_hat = torch.sparse.FloatTensor(indices, values, A_hat.size())
    return A_hat, coo_matrix(A.detach().cpu().numpy())

def psearch(n_train,emb,K,reg,seed):
    test = datasets.get('test',n=500,seed=seed)
    train = datasets.get('train',n=n_train,seed=seed)

    adj1 = torch.from_numpy(nx.adjacency_matrix(g1).todense())
    adj2 = torch.from_numpy(nx.adjacency_matrix(g2).todense())
    source_A_hat, _ = Laplacian_graph(adj1.numpy())
    target_A_hat, _ = Laplacian_graph(adj2.numpy())
    source_A_hat = source_A_hat.to_dense()
    target_A_hat = target_A_hat.to_dense()
    # source_A_hat = torch.from_numpy(adj1.numpy())
    # target_A_hat = torch.from_numpy(adj2.numpy())

    adj1 = adj1.cuda(1)
    adj2 = adj2.cuda(1)
    source_A_hat = source_A_hat.cuda(1)
    target_A_hat = target_A_hat.cuda(1)
    
    source_feats = torch.from_numpy(emb[:adj1.shape[0]]).type(torch.float).cuda(1)
    target_feats = torch.from_numpy(emb[adj1.shape[0]:]).type(torch.float).cuda(1)
    
    source_edgeindex = nx.to_scipy_sparse_matrix(g1).nonzero()
    source_edgeindex = torch.from_numpy(np.asarray([source_edgeindex[0], source_edgeindex[1]])).cuda(1)
    # print(source_edgeindex)

    target_edgeindex = nx.to_scipy_sparse_matrix(g2).nonzero()
    # print(target_edgeindex)
    target_edgeindex = torch.from_numpy(np.asarray([target_edgeindex[0], target_edgeindex[1]])).cuda(1)
    # print(target_edgeindex)

    gt_train = np.asarray([[k, v - adj1.shape[0]] for k,v in train])
    # print("gt_train: ", gt_train)

    print(source_feats.shape, target_feats.shape, source_A_hat.shape, target_A_hat.shape)
    model = gat_semisup_training(gt_train, source_feats, target_feats, source_A_hat, target_A_hat, source_edgeindex, target_edgeindex)

    source_outputs = model(source_edgeindex, 's', A_hat=source_A_hat)
    target_outputs = model(target_edgeindex, 't', A_hat=target_A_hat)
    # print(source_outputs)

    emb_gat = torch.vstack((source_outputs[-1], target_outputs[-1])).detach().cpu().numpy()
    # print(emb_gat)
    emb = np.concatenate((emb,emb_gat),axis=-1)

    traindata = []
    for k,v in train:
        traindata.append([emb[k],emb[v]])
    traindata = np.array(traindata)

    testdata = []
    for k,v in test:
        testdata.append([emb[k],emb[v]])
    testdata = np.array(testdata)
    
    zx,zy=align_cca(traindata,testdata,K=K,reg=reg)
    
    sim_matrix = get_sim(zx,zy,top_k=30)
    score=[]
    for top_k in [1,3,5,10, 15, 20, 25, 30]:
        score_ = hit_precision(sim_matrix,top_k=top_k)
        score.append(score_)
    return score

def linkpred_loss(embedding, A):
    pred_adj = torch.matmul(F.normalize(embedding), F.normalize(embedding).t())
    pred_adj = F.normalize((torch.min(pred_adj, torch.Tensor([1]).cuda(1))), dim = 1)
    # pred_adj = F.normalize((torch.min(pred_adj, torch.Tensor([1]))), dim = 1)
    #linkpred_losss = (pred_adj - A[index]) ** 2
    # print(pred_adj.shape, A.shape, pred_adj, A)
    linkpred_losss = (pred_adj - A) ** 2
    linkpred_losss = linkpred_losss.sum() / A.shape[1]
    return linkpred_losss

def gat_semisup_training(gt_train, source_feats, target_feats, source_A_hat, target_A_hat, source_edgeindex, target_edgeindex):
    lr = 0.005
    epochs = 50
    num_GCN_blocks = 2
    embedding_dim = 400
    num_heads_per_layer = [8 for i in range(1, num_GCN_blocks)] + [1]
    num_features_per_layer = [source_feats.shape[1]] + [embedding_dim for i in range(num_GCN_blocks)]
    model = GAT(
            num_of_layers = num_GCN_blocks,
            num_heads_per_layer=num_heads_per_layer,#[8, 8, 1],
            num_features_per_layer=num_features_per_layer,#[source_feats.shape[1], self.args.embedding_dim, self.args.embedding_dim, self.args.embedding_dim],
            add_skip_connection=False,
            source_feats = source_feats,
            target_feats = target_feats
        )
    # train = np.asarray([[x,y] for (x,y) in gt_train.items()])

    model = model.cuda(1)
    model.train()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad() 
        print("GAT learning epoch: {}".format(epoch))
        
        # print("source_A_hat", source_A_hat)
        source_outputs = model(source_edgeindex, 's', A_hat=source_A_hat)
        target_outputs = model(target_edgeindex, 't', A_hat=target_A_hat)
        # output = torch.cat((source_output, target_output), dim = 0)
        left = gt_train[:,0]
        right = gt_train[:,1]
        print('test here')
        left_x1 = source_outputs[-1][left]
        right_x1 = target_outputs[-1][right]
        sup_loss1 = (left_x1 - right_x1) ** 2
        sup_loss1 = sup_loss1.mean()
        left_x2 = source_outputs[-2][left]
        right_x2 = target_outputs[-2][right]
        sup_loss2 = (left_x2 - right_x2) ** 2
        sup_loss2 = sup_loss2.mean()
        sup_loss = sup_loss1 + sup_loss2
        unsup_loss_source = linkpred_loss(source_outputs[-1], source_A_hat) + linkpred_loss(source_outputs[-2], source_A_hat)
        unsup_loss_target = linkpred_loss(target_outputs[-1], target_A_hat) + linkpred_loss(target_outputs[-2], target_A_hat)
        unsup_loss = unsup_loss_source + unsup_loss_target
        loss = sup_loss + unsup_loss
        loss.backward()
        #loss = unsup_loss
        print('recent loss: {:.4f}, sup: {}, unsup: {:.4f}'.format(loss, sup_loss, unsup_loss))
        optimizer.step()
    model.eval()
    return model

anchors = dict(json.load(open('../data/dblp/anchors.txt','r')))
g1,g2 = pickle.load(open('../data/dblp/networks','rb'))
datasets = dataset(anchors)
n_anchors = len(anchors.keys())
# pool=Pool(min(16,os.cpu_count()-2))
# adj1 = nx.adjacency_matrix(g1)
# adj2 = nx.adjacency_matrix(g2)

if __name__ == '__main__':
    result=[]
    i = 0.8
    for seed in [2]: #range(3):
        d = 100
        fname = '../emb/emb_dblp_seed_{}_dim_{}'.format(seed,d)
        emb_c,emb_w,emb_t,emb_s = pickle.load(open(fname,'rb'))

        emb_attr = np.concatenate((emb_c,emb_w,emb_t),axis=-1)
        emb_all = np.concatenate((emb_c,emb_w,emb_t,emb_s),axis=-1)
        for model in [2]:
            n_train = int(i * n_anchors)
            emb = [emb_attr,emb_s,emb_all][model]
            model_name = ['MAUIL-a','MAUIL-s','MAUIL'][model]
            dim = emb.shape[-1]
            for K in [[0],[0],[80]][model]:
                for reg in [1000]: # [100,1000]:
                    score=[]
                    seed_ = list(range(10))
                    score_10 = psearch(n_train, emb, K, reg, seed_)
                    # print(score_10)
                    # score_10 = pool.map(partial(psearch,n_train,emb,K,reg),seed_)
                    # score_10 = np.array(score_10)
                    # assert score_10.shape==(10,4)
                    # score = np.mean(score_10,axis=0)
    
                    record = [i,seed,d,model_name,n_train,K,reg]+score_10
                    result.append(record)
                    print(record)

    json.dump(result,open(f'result_MAUIL_dblp_gat_{i}.txt','w'))
#
#
#