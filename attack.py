import os
import os.path as osp
import pickle
import random
import json
import argparse
import copy
import itertools

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

class AttackerStep:
    '''
    Generic class for attacker steps, under perturbation constraints
    specified by an "origin input" and a perturbation magnitude.
    Must implement project, step, and random_perturb
    '''
    def __init__(self, orig_input, B, eta, delta, alpha, C=None, use_grad=True):
        '''
        Initialize the attacker step with a given perturbation magnitude.
        Args:
            eps (float): the perturbation magnitude
            orig_input (ch.tensor): the original input
        '''
        self.orig_input = orig_input # torch type
        self.B = B
        self.eta = eta
        self.delta = delta
        self.alpha = alpha 
        self.C = C
        self.use_grad = use_grad


    def project(self, s,radius=1.):
        '''
        Given an input s, project it back into the feasible set
        Args:
            ch.tensor s : the input to project back into the feasible set.
        Returns:
            A `ch.tensor` that is the input projected back into
            the feasible set, that is,
        .. math:: \min_{s' \in S} \|s' - s\|_2
        '''
        raise NotImplementedError

    def step(self, s, g):
        '''
        Given a gradient, make the appropriate step according to the
        perturbation constraint (e.g. dual norm maximization for :math:`\ell_p`
        norms).
        Parameters:
            g (ch.tensor): the raw gradient
        Returns:
            The new input, a ch.tensor for the next step.
        '''
        raise NotImplementedError

    def random_perturb(self, s):
        '''
        Given a starting input, take a random step within the feasible set
        '''
        raise NotImplementedError

class BlackBoxStep(AttackerStep): 
    def __init__(self, orig_input, B, eta, delta, alpha, attack_loss,C=None, costVector = None, use_grad=False,cuda=True):
        super( BlackBoxStep, self).__init__(orig_input, B, eta, delta, alpha, C, use_grad)
        self.attack_loss = attack_loss
        self.N = orig_input.numel()
        self.cuda=cuda
        self.costVector = costVector
        self.vt = torch.zeros(self.N)

    def project(self, s,radius=1.):
        s = s.unsqueeze(0)
        s = s.renorm(p=1, dim=0, maxnorm=radius*self.B)
        if self.C != None:
            s = s * self.costVector
            s = s.renorm(p=1, dim=0, maxnorm=radius*self.C)
        s = s.view(-1)
        return s

    def step(self, s, g): # v_{t+1}
        update = s - self.eta * g
        update = self.project(update,1-self.alpha)
        return update

    def random_perturb(self):
        top_B = random.sample(range(self.N),self.B)
        st = torch.zeros(self.N)
        st[top_B] = 1
        return st

    def randomUnitVector(self):
        vec = np.array([np.random.normal(0., 1.) for i in range(self.N)])
        mag = np.linalg.norm(vec)
        return vec / mag

    def Bandit_step(self,node):
        with torch.no_grad():
            if self.cuda:
                u = torch.from_numpy(self.randomUnitVector()).cuda()
            else:
                u = torch.from_numpy(self.randomUnitVector())
            self.vt = self.vt + self.delta * u
            top_B = self.vt.sort(descending=True).indices[:self.B]
            st = torch.zeros(self.N)
            st[top_B] = 1
            L = self.attack_loss(st,node,0)
            sign=1
            coef=self.N/self.delta

            gradEst = coef*sign*L* u

            self.vt = self.step(self.vt,gradEst)
            return st

