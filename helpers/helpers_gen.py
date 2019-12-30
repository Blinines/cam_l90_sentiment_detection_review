# -*- coding: utf-8 -*-
import random
import numpy as np
from decimal import Decimal
from itertools import permutations
from copy import deepcopy

""" Mostly helpers for statistical tests : sign test and permutation test """

def get_variance(l):
    # Computing variance of list
    mean_val = np.mean(l)
    sum_val = [None] * len(l)
    for index, val in enumerate(l):
        sum_val[index] = (val - mean_val)**2
    return np.sum(sum_val)
    

def sign_test(y_1, y_2, y_true):
    # plus : clf_1 better than clf_2
    # minus : clf_2 better than clf_1
    # null : clf_1 and clf_2 predicted the same
    numbers = {'plus': 0, 'minus': 0, 'null': 0}

    for i in range(len(y_1)):
        if ((y_1[i] == y_true[i]) and (y_2[i] != y_true[i])):
            numbers['plus'] += 1
        elif ((y_1[i] != y_true[i]) and (y_2[i] == y_true[i])):
            numbers['minus'] += 1
        else:
            numbers['null'] +=1
    
    return numbers


def binomial(n, k):
    if 0 <= k <= n:
        num, denom = 1, 1
        for t in range(1, min(k, n - k) + 1):
            num *= n
            denom *= t
            n -= 1
        return num // denom
    else:
        return 0


def p_value_sign_test(numbers, q):
    N = 2*int(numbers['null']/2) + numbers['plus'] + numbers['minus']
    k = int(numbers['null']/2) + min(numbers['plus'], numbers['minus'])

    res = 0
    for i in range(k+1):
        res += Decimal(binomial(N, i)) * Decimal((q**i)) * Decimal(((1-q)**(N-i)))
    return float(2*res)



def re_sample(y_1, y_2, perm):
    y_1_new = deepcopy(y_1)
    y_2_new = deepcopy(y_2)
    for index, elt in enumerate(perm):
        if elt == 1:
            y_1_new[index], y_2_new[index] = y_2_new[index], y_1_new[index]
    return y_1_new, y_2_new


def p_value_permutation_test(y_1, y_2, y_true, r=5000):
    ''' r: number of permutations used '''

    # init variables
    abs_diff_map = abs(np.mean(y_1==y_true) - np.mean(y_2==y_true))
    s = 0  # number of permuted samples with absolute diff in mean >= original one 
 
    # choosing permutations to use
    n = len(y_1)
    for _ in range(r):
        perm = np.array([0]*int(n/2) + [1]*int(n/2))
        np.random.shuffle(perm)
        y_1_sampled, y_2_sampled = re_sample(y_1=y_1, y_2=y_2, perm=perm)
        if abs(np.mean(y_1_sampled==y_true) - np.mean(y_2_sampled==y_true)) >= abs_diff_map:
            s += 1
    
    return float(s+1)/(r+1)




