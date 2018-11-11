# -*- coding: utf-8 -*-
"""
Created on Sun July  27 00:32:00 2018

@author: Rudrajit
"""
import torch
import torch.nn as nn
import torch.nn.functional as F



class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L = 64
        self.D = 32
        self.K = 1

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )


    def combine_feats(self, feats_all):
        # size of feats_all = [64,256]
        w = torch.randn((256, 1), requires_grad=True)
        feats_comb = torch.mm(feats_all, w)
        feats_comb_relu = F.relu(feats_comb)
        return feats_comb_relu


    def combine_feats_all(self, patches_feats_all):
        #size of patches_feats_all = [training_examples,64,256]
        patches_feats_comb = torch.zeros((48, 64))
        for i in range(48):
            patches_feats_comb[i,:] = self.combine_feats(torch.squeeze(patches_feats_all[i,:,:]))

        return patches_feats_comb


    def forward(self, patches):
        patches_feats_comb = self.combine_feats_all(patches_feats_all) # N(=48)XL(=64)
        patches_att = self.attention(patches_feats_comb)  # NxK
        patches_att = torch.transpose(patches_att, 1, 0)  # KxN
        patches_att_wts = F.softmax(patches_att, dim=1)  # softmax over N

        weighted_feat = torch.mm(patches_att_wts, patches_feats_comb)  # KxL

        Y_prob = self.classifier(weighted_feat)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, patches_att_wts
        #return Y_prob, patches_att_wts

    
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data[0]

        return error, Y_hat


     def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A