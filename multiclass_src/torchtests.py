import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random

from mc_torchconfusion import *

DEVICE = 'cpu'

class TestTorchConfusion(unittest.TestCase):

    def test_grad_l_tp_0_0(self):
        """ Testing TP 
        - Should be no TP's as there are no PT or GT. 
        - These should all non-TP's as they're not positive.
        - Gradient should be almost 0, as it doesn't go anywhere. 
        """
        gt = torch.tensor([0.0, 0.0])
        pt = torch.tensor([0.0, 0.0], requires_grad=True)
        thresholds = torch.tensor([0.1, 0.5, 0.9])

        inputs = (gt, pt, thresholds)

        # gradient here is 0, because they don't go anywhere. 
        test = torch.autograd.gradcheck(l_tp, inputs, eps=1e-6, atol=1e-4, raise_exception=True)
        jacob = torch.autograd.functional.jacobian(l_tp, inputs=inputs)

        print(jacob)
        print(test)

        res = l_tp(gt, pt, thresholds)
        # res.backward(torch.ones_like(pt))
        res.sum().backward()
        print(pt.grad.data)


    def test_grad_l_tp_1_0(self):
        """ 
        """
        gt = torch.tensor([1.0, 1.0, 1.0])
        pt = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
        thresholds = torch.tensor([0.1, 0.5, 0.9])

        inputs = (gt, pt, thresholds)

        test = torch.autograd.gradcheck(
            l_tp, inputs, eps=1e-2, atol=1e-2, raise_exception=True)

        res = l_tp(gt, pt, thresholds)
        res.backward(torch.ones_like(pt))
        print(pt.grad)

    def test_grad_l_fp_1_0(self):
        """ 
        """
        gt = torch.tensor([1.0, 1.0, 1.0])
        pt = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        thresholds = torch.tensor([0.1, 0.5, 0.9])

        inputs = (gt, pt, thresholds)

        # gradient here is 0, because they don't go anywhere.
        torch.autograd.gradcheck(
            l_tp, inputs, eps=1e-2, atol=1e-2, raise_exception=True)



    def test_ordering_correctness(self):
        def make_orderings(num_tensor):
            '''
            Returns a tensor with entries 0, 0.5 and 1 based on how
            the adjacent numbers in the tensor num_tensor compare.
            num_tensor: 2D array of numbers
            '''
            dim1, dim2 = num_tensor.shape[0], num_tensor.shape[1]
            orderings = torch.empty((dim1, dim2))
            for i in range(dim1-1):
                for j in range(dim2):
                    orderings[i][j] = 0.0 if num_tensor[i][j] < num_tensor[i+1][j] \
                            else (0.5 if num_tensor[i][j] == num_tensor[i+1][j] else 1.0)
                    orderings[dim1-1][j] = 0.0
            return orderings

        def test_ordering(_pred, thresh):
            '''
            _pred, thresh: tensors containing predictions, thresholds
            This function checks if the output of the predictions are in the right order
            and on the right side of the thresholds.
            '''
            thresh = torch.where(thresh == 0.0, torch.tensor([0.01], device=DEVICE),
                                 torch.where(thresh == 1.0, torch.tensor([0.99], device=DEVICE), thresh))
            pred = torch.reshape(torch.repeat_interleave(_pred, \
                thresh.shape[0]), (-1, thresh.shape[0]))
            outputs = torch.where(pred < thresh, torch.tensor([0.0]),\
                torch.where(pred == thresh, torch.tensor([0.5]), torch.tensor([1.0])))
            comparisons = make_orderings(pred)
            # Concatenate the tensors for easier comparing later
            truth_board = torch.cat((outputs, comparisons), 1)
            # both pred and thresh should be shape: [batch_size, thresholds]
            threshs = thresh.repeat(pred.shape[0],).reshape(pred.shape[0], -1)
            test_results = heaviside_agg(pred, threshs, None)
            test_outputs = torch.where(test_results < 0.5, torch.tensor([0.0]), \
                torch.where(test_results == 0.5, torch.tensor([0.5]), torch.tensor([1.0])))
            test_comparisons = make_orderings(test_results)
            test_board = torch.cat((test_outputs, test_comparisons), 1)

            torch.testing.assert_allclose(truth_board, test_board)
        
        # Comparing for different predictions and thresholds
        small_delta = 0.01
        thresholds = torch.tensor([random.uniform(0.2, 0.8) for i in range(3)])
        pred = []
        for i in range(3):
            pred.append(max(0.0, thresholds[i]-small_delta))
            pred.append(min(1.0, thresholds[i]+small_delta))
            # Also a random number in region m1, m2 and m3
            tt = torch.min(thresholds[i], 1-thresholds[i])
            pred.append(random.uniform(0.0, thresholds[i]-tt/2))
            pred.append(random.uniform(thresholds[i]-tt/2, thresholds[i]+tt/2))
            pred.append(random.uniform(thresholds[i]+tt/2, 1.0))
        pred = torch.tensor(pred)
        pred, indices = torch.sort(pred)
        test_ordering(pred, thresholds)


if __name__ == '__main__':
    unittest.main()
