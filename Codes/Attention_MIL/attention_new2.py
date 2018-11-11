# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 00:40:44 2018

@author: Rudrajit Das
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L = 3072
        self.D = 512#1024#512
        self.K = 1

        self.attention1 = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
            #nn.Linear(self.D, self.K)
        )
        
        self.attention2 = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
            #nn.Linear(self.D, self.K)
        )
        
        self.attention3 = nn.Sequential(
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )


    def forward(self, patches_feats_comb):
        patches_feats_comb = patches_feats_comb.squeeze(0)
        # patches_feats_comb - N(=48)XL(=1024)
        patches_att1 = self.attention1(patches_feats_comb)  # NxK
        patches_att2 = self.attention2(patches_feats_comb)  # NxK
        patches_att_gated = torch.mul(patches_att1, patches_att2)
        #patches_att_gated = patches_att_gated.squeeze(0)
        #m = nn.Dropout(p=0.5)
        #patches_att_gated = m(patches_att_gated)
        patches_att = self.attention3(patches_att_gated)
        
        patches_att = patches_att.squeeze(0)
        #print(patches_att.size())
        patches_att = torch.transpose(patches_att, 1, 0)  # KxN
        patches_att_wts = F.softmax(patches_att, dim=1)  # softmax over N
        
        
        #print(patches_att_wts.size())
        #print(patches_feats_comb.size())
        weighted_feat = torch.mm(patches_att_wts, patches_feats_comb)  # KxL
        
        #m = nn.Dropout(p=0.2)
        #weighted_feat = m(weighted_feat)
        #print(weighted_feat.shape)
        Y_prob = self.classifier(weighted_feat)
        #weighted_feat_with_bias = torch.ones([1,1025])
        #weighted_feat_with_bias[:,0:1024] = weighted_feat
        #weighted_feat_with_bias = weighted_feat_with_bias.cuda()
        #Y_prob = self.classifier(weighted_feat_with_bias)
        
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