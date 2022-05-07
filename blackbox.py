import os
import os.path as osp
import pickle
import random
import json
import argparse
import copy
import itertools
import importlib

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

from torch.utils.data import DataLoader

from utils.utils import load_citation,accuracy,sparse_mx_to_torch_sparse_tensor,sgc_precompute,LoadGraphDataSet

from utils.normalization import fetch_normalization, row_normalize

from attack import BlackBoxStep


class BlackboxAttackProcedure_NodeClassification(object): 
    def __init__(self,config,cuda=False):
        self.T = config["queriesNumber"]
        self.cuda = cuda
        if self.cuda:
            self.device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')
        else:
            self.device = torch.device('cpu') #
        adj, features, labels, idx_train, idx_val, idx_test,graph = load_citation(config["dataset"],cuda=cuda)
        self.x = features
        self.y = labels
        self.tensor_adjacency = adj
        self.idx_test = idx_test
        self.graph = graph

        self.eta = config["eta"]
        self.delta = config["delta"]
        self.alpha = config["alpha"]
        self.B = config["B"]
        self.C = None

        self.targetNodeSet = pickle.load(open(config["attackSet"], "rb"))
        if config["model"] == "GCN":
            module = importlib.import_module(config["model_definition_file"])
            GCN = getattr(module,config["model"])
            checkpoint = torch.load(config["model_path"])
            model = GCN(config["nfeat"],config["nhid"],config["nclass"],config["dropout"])
            model.load_state_dict(checkpoint)
            self.target_model = model
            self.target_model_name = "GCN"
        elif config["model"] == "SGC":
            module = importlib.import_module(config["model_definition_file"])
            SGC = getattr(module,config["model"])
            checkpoint = torch.load(config["model_path"])
            model = SGC(nfeat=config["nfeat"],nclass=config["nclass"])
            model.load_state_dict(checkpoint)
            self.target_model = model
            self.target_model_name = "SGC"

    @staticmethod
    def build_adjacency(adj_dict):
        """create adjacent matric based on adjacent list"""
        edge_index = []
        num_nodes = len(adj_dict)
        for src, dst in adj_dict.items():
            edge_index.extend([src, v] for v in dst)
            edge_index.extend([v, src] for v in dst)
        # delete the duplicated edges
        edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))
        edge_index = np.asarray(edge_index)
        adjacency = sp.coo_matrix((np.ones(len(edge_index)), 
                                   (edge_index[:, 0], edge_index[:, 1])),
                    shape=(num_nodes, num_nodes), dtype="float32")
        return adjacency

    def perturb(self,sv,node): # return perturbed graph adj
        old_edges = self.graph[node]
        perturbed_edges = np.argwhere(sv.numpy()==1).flatten().tolist()
        total_edges = old_edges + perturbed_edges
        common_edges = [edge for edge in old_edges if edge in perturbed_edges]
        new_edges = [edge for edge in total_edges if edge not in common_edges]
        graph = copy.deepcopy(self.graph)
        graph[node] = new_edges
        adj = self.build_adjacency(graph)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj_normalizer = fetch_normalization("AugNormAdj")
        adj = adj_normalizer(adj)
        adj = sparse_mx_to_torch_sparse_tensor(adj).float()
        if self.cuda:
            adj = adj.cuda()
        return adj

    def queryBox(self,sv,node):
        adjacency = self.perturb(sv,node)
        if self.target_model_name == "GCN":
            return self.target_model(self.x, adjacency)[node].cpu().detach().numpy()
        elif self.target_model_name == "SGC":
            perturb_feature,_ = sgc_precompute(self.x,adjacency,2)
            return self.target_model(perturb_feature[node]).cpu().detach().numpy()

    def attackLoss(self,sv,node,kappa):
        query_result = self.queryBox(sv,node)
        mask = np.ones(query_result.size,dtype=np.bool)
        mask[self.y[node]] = False
        loss = query_result[self.y[node]] - query_result[mask].max()
        return max(loss,-kappa)

    def banditAttack(self,T,node):
        adjVector_node = torch.zeros(len(self.graph))
        adjVector_node[self.graph[node]] = 1
        step = BlackBoxStep(adjVector_node,self.B, self.eta, self.delta, self.alpha, self.attackLoss,use_grad=False,cuda=self.cuda)

        for t in range(T):
            perturbation = step.Bandit_step(node)
            query_result = self.queryBox(perturbation,node)
            if query_result.argmax()!=self.y[node]:
                print("Bandit attack successfully",t+1,node)
                return 1,t+1
        print("Bandit attack failed",T,node)
        return 0,T

    def randomAttack(self,T,node):
        adjVector_node = torch.zeros(len(self.graph))
        adjVector_node[self.graph[node]] = 1
        step = BlackBoxStep(adjVector_node,self.B, self.eta, self.delta, self.alpha, self.attackLoss,use_grad=False,cuda=self.cuda)

        for t in range(T):
            perturbation = step.random_perturb()
            query_result = self.queryBox(perturbation,node)
            if query_result.argmax()!=self.y[node]:
                print("Random attack successfully",t+1,node)
                return 1,t+1
        print("Random attack failed",T,node)
        return 0,T

    def attack(self):
        ret = self.banditAttack(self.T,random.choice(self.targetNodeSet))
        ret = self.randomAttack(self.T,random.choice(self.targetNodeSet))

class BlackboxAttackProcedure_GraphClassification(object): 
    def __init__(self,config,cuda=False):
        self.T = config["queriesNumber"]
        self.cuda = cuda
        if self.cuda:
            self.device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')
        else:
            self.device = torch.device('cpu') #
        self.eta = config["eta"]
        self.delta = config["delta"]
        self.alpha = config["alpha"]
        self.B = config["B"]
        self.C = None


        self.targetGraphSet = pickle.load(open(config["attackSet"], "rb"))
        dataset = LoadGraphDataSet(config["dataset"])
        test_loader = DataLoader(dataset.test, batch_size=1, shuffle=False, drop_last=False, collate_fn=dataset.collate)
        self.graphs = list(test_loader)
        self.testset = dataset.test
        if config["model"] == "GIN":
            module = importlib.import_module(config["model_definition_file"])
            GIN = getattr(module,config["model"])
            checkpoint = torch.load(config["model_path"],map_location=torch.device('cpu'))
            net_params = config['net_params']
            model = GIN(net_params)
            model.load_state_dict(checkpoint)
            self.target_model = model
            self.target_model_name = "GIN"

    def perturb(self,graphid,S):
        A = self.testset.Adj_matrices[graphid]
        graph,label = self.graphs[graphid]
        graph2 = copy.deepcopy(graph)
        num_nodes = graph2.number_of_nodes()
        for i in range(num_nodes):
            for j in range(num_nodes):
                if S[i,j]==1:
                    if graph2.has_edge_between(i,j):
                        graph2.remove_edges(graph2.edge_ids(i,j))
                    else:
                        graph2.add_edge(i,j)
                        graph2.edata["feat"][-1] = A[i,j]
        return graph2

    def queryBox(self,graphid,S):
        graph = self.perturb(graphid,S)
        g = graph.to(self.device)
        h = graph.ndata['feat'].to(self.device)
        e = graph.edata['feat'].to(self.device)
        logits = self.target_model(g,h,e)
        return logits.cpu().detach().numpy()[0]

    def attackLoss(self,sv,graphid,kappa):
        graph,label = self.graphs[graphid]
        num_nodes = graph.number_of_nodes()
        sv = sv.reshape((num_nodes,num_nodes))
        query_result = self.queryBox(graphid,sv)
        mask = np.ones(query_result.size,dtype=np.bool)
        mask[label.item()] = False
        loss = query_result[label.item()] - query_result[mask].max()
        return max(loss,-kappa)

    def banditAttack(self,T,graphid):
        graph,label = self.graphs[graphid]
        num_nodes = graph.number_of_nodes()
        targetgraph_adj = self.testset.Adj_matrices[graphid]
        targetgraph_adj = torch.from_numpy(targetgraph_adj)
        step = BlackBoxStep(targetgraph_adj,self.B, self.eta, self.delta, self.alpha, self.attackLoss,use_grad=False,cuda=self.cuda)

        for t in range(T):
            perturbation = step.Bandit_step(graphid)
            perturbation = perturbation.reshape((num_nodes,num_nodes))
            query_result = self.queryBox(graphid,perturbation)
            if query_result.argmax()!=label.item():
                print("Bandit attack successfully",t+1,graphid)
                return 1,t+1
        print("Bandit attack failed",T,graphid)
        return 0,T

    def randomAttack(self,T,graphid):
        graph,label = self.graphs[graphid]
        num_nodes = graph.number_of_nodes()
        targetgraph_adj = self.testset.Adj_matrices[graphid]
        targetgraph_adj = torch.from_numpy(targetgraph_adj)
        step = BlackBoxStep(targetgraph_adj,self.B, self.eta, self.delta, self.alpha, self.attackLoss,use_grad=False,cuda=self.cuda)

        for t in range(T):
            perturbation = step.random_perturb()
            perturbation = perturbation.reshape((num_nodes,num_nodes))
            query_result = self.queryBox(graphid,perturbation)
            if query_result.argmax()!=label.item():
                print("Random attack successfully",t+1,graphid)
                return 1,t+1
        print("Random attack failed",T,graphid)
        return 0,T

    def attack(self):
        ret = self.banditAttack(self.T,random.choice(self.targetGraphSet))
        ret = self.randomAttack(self.T,random.choice(self.targetGraphSet))




def main(config):
	# attacker = BlackboxAttackProcedure_NodeClassification(config)
	# attacker.attack()
	attacker = BlackboxAttackProcedure_GraphClassification(config)
	attacker.attack()


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Blackbox attack against GNN')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-l', '--log', default='results.txt', type=str,
                        help='logname')
    args = parser.parse_args()

    config = json.load(open(args.config))

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    print(config)
    main(config)