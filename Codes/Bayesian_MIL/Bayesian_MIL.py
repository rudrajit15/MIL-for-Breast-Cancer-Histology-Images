# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 11:04:41 2018

@author: Rudrajit
"""

import numpy as np
import os
import math
import scipy as sc
#from keras.datasets import mnist

from scipy.stats import logistic
# x is a NK*d matrix (d = no. of features, N = no. of images, K = no. of patches)
# y is a N*1 matrix
def Bayesian(x_inp,y_inp):
    x = x_inp
    y = y_inp
    d = x.shape[1]
    N = y.shape[0]
    K = (int)(x.shape[0]/N)
    alpha = np.ones(d)
    w = np.zeros([d,1])
    #w_remove = np.zeros([d,1])
    tau = 1e12
    eta = 2.5*1e-1
    #eta = 8*1e-4
    epsilon1 = 1e-5
    #epsilon1 = 0.003
    
    epsilon2 = 1e-3
    #epsilon2 = 0.3
    alpha_min = 1e-5
    iters = 0
    err_prev = 1e6
    
    while True:
        # Feature removal part
        remove = np.where(alpha > tau)
        alpha = np.delete(alpha,remove)
        w = np.delete(w,remove,0)
        x = np.delete(x,remove,1)
        d = x.shape[1]
        print(d)
        H_map_inv = np.zeros([d,d])
        while True:
            g = np.zeros([d,1])
            s = logistic.cdf(np.dot(x,w))
            #print(s.shape)
            t = np.multiply(x,s)
            A = np.diag(alpha)
            for i in range(0,N):
                si = 1-s[i*K:(i+1)*K]
                pi = 1-np.prod(si)
                beta_i = (1-pi)/pi
                ti = np.reshape(np.sum(t[i*K:(i+1)*K,:],0),[d,1])
                g = g + (y[i]*beta_i-(1-y[i]))*ti
            g = g - np.dot(A,w)
            
            #changed
            g = -g
            
            H = -A
            s_minus = logistic.cdf(np.dot(x,-w))
            s2 = np.multiply(s,s_minus)
            t2 = np.multiply(x,s2)
            for i in range(0,N):
                si = 1-s[i*K:(i+1)*K]
                pi = 1-np.prod(si)
                beta_i = (1-pi)/pi
                ui = np.dot(np.transpose(x[i*K:(i+1)*K,:]),t2[i*K:(i+1)*K,:])
                H = H + (y[i]*beta_i-(1-y[i]))*ui
                ti = np.reshape(np.sum(t[i*K:(i+1)*K,:],0),[d,1])
                H = H - y[i]*beta_i*(beta_i+1)*np.dot(ti,np.transpose(ti))
            
            #changed
            H = -H
            
            H_map_inv = np.linalg.inv(H)   
            #print(np.max(np.diag(H)))
            
            err_curr = (np.linalg.norm(g)/d)
            print(err_curr)
            
            #if err_curr < err_prev:
            #    err_prev = err_curr
            #    if err_curr < epsilon1:
            #        break
            #    else:
            #       w = w - eta*np.dot(H_map_inv,g) 
            #else:
            #   eta = eta*0.85
            
            if err_curr < epsilon1:
                break
            else:
                w = w - eta*np.dot(H_map_inv,g)
                
        #break
                
        sigma = np.reshape(np.diag(H_map_inv),[d,1])
        alpha2 = np.reshape(np.divide(1,np.multiply(w,w)+(sigma)),[d,])
        print(alpha2)
        #alpha2[alpha2 < 0] = alpha_min
        
        print(np.max(np.abs(np.log(alpha2))-np.log(alpha)))
        
        if np.max(np.abs(np.log(alpha2))-np.log(alpha)) < epsilon2:
            break
        else:
            alpha = alpha2
        
        iters = iters+1
        print(iters)
        
        if(iters > 10):
            break
        
    return x,w


# x is a NK*d' matrix (d' = new no. of features, N = no. of images, K = no. of patches)
# w is a d'*1 vector
# y_pred is a NK*1 matrix
def get_new_labels(x,w):
    y_pred = logistic.cdf(np.dot(x,w))
    tmp = np.reshape(y_pred,[y_pred.shape[0],])
    thr = sum(tmp)/y_pred.shape[0]
    #thr = 0.0123
    #y_pred[y_pred < thr] = 0
    #y_pred[y_pred >= thr] = 1
    #y_pred = y_pred.astype(int)
    return y_pred