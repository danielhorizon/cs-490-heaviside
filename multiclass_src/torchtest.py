#!/usr/bin/env python3

#import pathlib
#ROOT_PATH = pathlib.Path(__file__).parent.parent.absolute()
#DATA_PATH = ROOT_PATH.joinpath('data').absolute()
#SRC_PATH = ROOT_PATH.joinpath('src').absolute()
#print(ROOT_PATH)
#import sys
#sys.path.append(str(SRC_PATH))

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random

from torchconfusion1 import *

# from confusion_vectorized import *

device = "cuda:0"
class TestTorchConfusion(unittest.TestCase):

    def test_l_tp_adj_0_0(self):
        ''' there should be no true positives as there are no positive pt or gt '''
        # ground truth
        gt = torch.tensor([0.0, 0.0, 0.0])
        # prediction
        pt = torch.tensor([0.0, 0.0, 0.0])
        # thresholds
        thresholds = torch.tensor([0.1, 0.5, 0.9])
        #agg = None
        tp = l_tp_adj(device, gt, pt, thresholds, approx=linear_approx())
        #print(tp, torch.zeros_like(tp))
        torch.testing.assert_allclose(tp, torch.zeros_like(tp))

    def test_l_tp_adj_1_0(self):
        ''' there should be no true positives as there are no positive pt even through there are positive gt'''
        # ground truth
        gt = torch.tensor([1.0, 1.0, 1.0])
        # prediction
        pt = torch.tensor([0.0, 0.0, 0.0])
        # thresholds
        thresholds = torch.tensor([0.1, 0.5, 0.9])
        #agg = None
        tp = l_tp_adj(device, gt, pt, thresholds, approx=linear_approx())
        torch.testing.assert_allclose(tp, torch.zeros_like(tp))

    def test_l_tp_adj_1_1(self):  # FAIL
        ''' there should be all true positives as there are all positive pt and positive gt'''
        # ground truth
        gt = torch.tensor([1.0, 1.0, 1.0])
        # prediction
        pt = torch.tensor([1.0, 1.0, 1.0])
        # thresholds
        thresholds = torch.tensor([0.1, 0.5, 0.9])
        #agg = None
        tp = l_tp_adj(device, gt, pt, thresholds, approx=linear_approx())
        print("test_l_tp_adj_1_1: {}".format(tp))
        # torch.ones_like(tp) # summing across 9 values in an array all of which are 1
        torch.testing.assert_allclose(tp, torch.tensor(3., device='cuda:0'))

    def test_l_tp_adj_0_1(self):
        ''' there should be no true positives as there are all positive pt but no positive gt'''
        # ground truth
        gt = torch.tensor([0.0, 0.0, 0.0])
        # prediction
        pt = torch.tensor([1.0, 1.0, 1.0])
        # thresholds
        thresholds = torch.tensor([0.1, 0.5, 0.9])
        #agg = None
        tp = l_tp_adj(device, gt, pt, thresholds, approx=linear_approx())
        torch.testing.assert_allclose(tp, torch.zeros_like(tp))

    def test_l_fn_0_0(self):
        '''Zero false negatives, all true negatives'''
        # ground truth
        gt = torch.tensor([0.0, 0.0, 0.0])
        # prediction
        pt = torch.tensor([0.0, 0.0, 0.0])
        # thresholds
        thresholds = torch.tensor([0.1, 0.5, 0.9])
        #agg = None
        fn = l_fn_adj(device, gt, pt, thresholds, approx=linear_approx())
        torch.testing.assert_allclose(fn, torch.zeros_like(fn))

    def test_l_fn_0_1(self):
        '''No labeled negatives, all false positives'''
        # ground truth
        gt = torch.tensor([0.0, 0.0, 0.0])
        # prediction
        pt = torch.tensor([1.0, 1.0, 1.0])
        # thresholds
        thresholds = torch.tensor([0.1, 0.5, 0.9])
        #agg = None
        fn = l_fn_adj(device, gt, pt, thresholds, approx=linear_approx())
        torch.testing.assert_allclose(fn, torch.zeros_like(fn))

    def test_l_fn_1_0(self):  # FAIL
        '''All false negatives'''
        # ground truth
        gt = torch.tensor([1.0, 1.0, 1.0])
        # prediction
        pt = torch.tensor([0.0, 0.0, 0.0])
        # thresholds
        thresholds = torch.tensor([0.1, 0.5, 0.9])
        #agg = None
        fn = l_fn_adj(device, gt, pt, thresholds, approx=linear_approx())
        # torch.ones_like(fn) # summing across 9 values in an array all of which are 1
        torch.testing.assert_allclose(fn, torch.tensor(9., device='cuda:0'))

    def test_1_fn_1_1(self):
        '''No false negatives, all true positives'''
        # ground truth
        gt = torch.tensor([1.0, 1.0, 1.0])
        # prediction
        pt = torch.tensor([1.0, 1.0, 1.0])
        # thresholds
        thresholds = torch.tensor([0.1, 0.5, 0.9])
        #agg = None
        fn = l_fn_adj(device, gt, pt, thresholds, approx=linear_approx())
        torch.testing.assert_allclose(fn, torch.zeros_like(fn))

    def test_l_fp_0_0(self):
        '''No false positives, all true negatives'''
        # ground truth
        gt = torch.tensor([0.0, 0.0, 0.0])
        # prediction
        pt = torch.tensor([0.0, 0.0, 0.0])
        # thresholds
        thresholds = torch.tensor([0.1, 0.5, 0.9])
        #agg = None
        fp = l_fp_adj(device, gt, pt, thresholds, approx=linear_approx())
        torch.testing.assert_allclose(fp, torch.zeros_like(fp))

    def test_l_fp_0_1(self):  # FAIL
        '''All false positives'''
        # ground truth
        gt = torch.tensor([0.0, 0.0, 0.0])
        # prediction
        pt = torch.tensor([1.0, 1.0, 1.0])
        # thresholds
        thresholds = torch.tensor([0.1, 0.5, 0.9])
        #agg = None
        fp = l_fp_adj(device, gt, pt, thresholds, approx=linear_approx())
        # debugging set:
        print(fp)
        print(torch.ones_like(fp))
        # torch.ones_like(fp) # summing across 9 values in an array all of which are 1
        torch.testing.assert_allclose(fp, torch.tensor(9., device='cuda:0'))

    def test_l_fp_1_0(self):
        '''No false positives, all false negatives'''
        # ground truth
        gt = torch.tensor([1.0, 1.0, 1.0])
        # prediction
        pt = torch.tensor([0.0, 0.0, 0.0])
        # thresholds
        thresholds = torch.tensor([0.1, 0.5, 0.9])
        #agg = None
        fp = l_fp_adj(device, gt, pt, thresholds, approx=linear_approx())
        torch.testing.assert_allclose(fp, torch.zeros_like(fp))

    def test_l_fp_1_1(self):
        '''No false positives, all true positives'''
        # ground truth
        gt = torch.tensor([1.0, 1.0, 1.0])
        # prediction
        pt = torch.tensor([1.0, 1.0, 1.0])
        # thresholds
        thresholds = torch.tensor([0.1, 0.5, 0.9])
        #agg = None
        fp = l_fp_adj(device, gt, pt, thresholds, approx=linear_approx())
        torch.testing.assert_allclose(fp, torch.zeros_like(fp))

    def test_l_tn_0_0(self):  # FAIL
        '''All true negatives'''
        # ground truth
        gt = torch.tensor([0.0, 0.0, 0.0])
        # prediction
        pt = torch.tensor([0.0, 0.0, 0.0])
        # thresholds
        thresholds = torch.tensor([0.1, 0.5, 0.9])
        #agg = None
        tn = l_tn_adj(device, gt, pt, thresholds, approx=linear_approx())
        #print(tn)
        # torch.ones_like(tn) # summing across 9 values in an array all of which are 1
        torch.testing.assert_allclose(tn, torch.tensor(9., device='cuda:0'))

    def test_l_tn_0_1(self):
        '''No true negatives, all false positives'''
        # ground truth
        gt = torch.tensor([0.0, 0.0, 0.0])
        # prediction
        pt = torch.tensor([1.0, 1.0, 1.0])
        # thresholds
        thresholds = torch.tensor([0.1, 0.5, 0.9])
        #agg = None
        tn = l_tn_adj(device, gt, pt, thresholds, approx=linear_approx())
        torch.testing.assert_allclose(tn, torch.zeros_like(tn))

    def test_l_tn_1_0(self):
        '''No true negatives, all false negatives'''
        # ground truth
        gt = torch.tensor([1.0, 1.0, 1.0])
        # prediction
        pt = torch.tensor([0.0, 0.0, 0.0])
        # thresholds
        thresholds = torch.tensor([0.1, 0.5, 0.9])
        #agg = None
        tn = l_tn_adj(device, gt, pt, thresholds, approx=linear_approx())
        torch.testing.assert_allclose(tn, torch.zeros_like(tn))

    def test_l_tn_1_1(self):
        '''No true negatives, all true positives'''
        # ground truth
        gt = torch.tensor([1.0, 1.0, 1.0])
        # prediction
        pt = torch.tensor([1.0, 1.0, 1.0])
        # thresholds
        thresholds = torch.tensor([0.1, 0.5, 0.9])
        #agg = None
        tn = l_tn_adj(device, gt, pt, thresholds, approx=linear_approx())
        torch.testing.assert_allclose(tn, torch.zeros_like(tn))

    # def test_ordering_correctness(self):

    #     def make_orderings(num_tensor):
    #         '''
    #         Returns a tensor with entries 0, 0.5 and 1 based on how
    #         the adjacent numbers in the tensor num_tensor compare.
    #         num_tensor: 2D array of numbers
    #         '''
    #         print(num_tensor)
    #         # dim1, dim2 = num_tensor.shape[0], num_tensor.shape[1]
    #         dim1, dim2 = num_tensor.shape[0], 1
    #         orderings = torch.empty((dim1, dim2))
    #         for i in range(dim1-1):
    #             for j in range(dim2):
    #                 orderings[i][j] = 0.0 if num_tensor[i][j] < num_tensor[i+1][j] \
    #                     else (0.5 if num_tensor[i][j] == num_tensor[i+1][j] else 1.0)
    #                 orderings[dim1-1][j] = 0.0
    #         return orderings

    #     def test_ordering(_pred, thresh):
    #         '''
    #         _pred, thresh: tensors containing predictions, thresholds
    #         This function checks if the output of the predictions are in the right order
    #         and on the right side of the thresholds.
    #         '''
    #         device="cuda:0"
    #         thresh=thresh.to(device)
    #         _pred = _pred.to(device)
    #         thresh = torch.where(thresh == 0.0, torch.tensor([0.01], device=device),
    #                              torch.where(thresh == 1.0, torch.tensor([0.99], device=device), thresh)).to(device)
    #         pred = torch.reshape(torch.repeat_interleave(_pred,
    #                                                      thresh.shape[0]), (-1, thresh.shape[0])).to(device)
    #         outputs = torch.where(pred < thresh, torch.tensor([0.0], device=device),
    #                               torch.where(pred == thresh, torch.tensor([0.5], device=device), torch.tensor([1.0], device=device))).to(device)
    #         comparisons = make_orderings(pred).to(device)
    #         # Concatenate the tensors for easier comparing later
    #         truth_board = torch.cat((outputs, comparisons), 1).to(device)
    #         # both pred and thresh should be shape: [batch_size, thresholds]
    #         threshs = thresh.repeat(pred.shape[0],).reshape(pred.shape[0], -1).to(device)
    #         # test_results = heaviside_sum(pred, threshs, None)
    #         test_results = heaviside_sum(pred, threshs, approx=linear_approx())
    #         test_outputs = torch.where(test_results < 0.5, torch.tensor([0.0], device=device),
    #                                    torch.where(test_results == 0.5, torch.tensor([0.5], device=device), torch.tensor([1.0], device=device)))
    #         test_comparisons = make_orderings(test_results)
    #         test_board = torch.cat((test_outputs, test_comparisons), 1)

    #         torch.testing.assert_allclose(truth_board, test_board)

    #     # Comparing for different predictions and thresholds
    #     small_delta = 0.01
    #     thresholds = torch.tensor([random.uniform(0.2, 0.8) for i in range(3)])
    #     pred = []
    #     for i in range(3):
    #         pred.append(max(0.0, thresholds[i]-small_delta))
    #         pred.append(min(1.0, thresholds[i]+small_delta))
    #         # Also a random number in region m1, m2 and m3
    #         tt = torch.min(thresholds[i], 1-thresholds[i])
    #         pred.append(random.uniform(0.0, thresholds[i]-tt/2))
    #         pred.append(random.uniform(thresholds[i]-tt/2, thresholds[i]+tt/2))
    #         pred.append(random.uniform(thresholds[i]+tt/2, 1.0))
    #     pred = torch.tensor(pred)
    #     pred, indices = torch.sort(pred)
    #     test_ordering(pred, thresholds)


if __name__ == '__main__':
    unittest.main()
