import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import warnings

from mc_torchconfusion import *

DEVICE = 'cpu'

class TestTorchConfusion(unittest.TestCase):
    def test_grad_l_tp_zero(self):
        """ Testing TP 
        - Should be no TP's as there are no PT or GT. These should all non-TP's as they're not positive.
        - Gradient should be almost 0, as it doesn't go anywhere. 
        """
        print("--- TP - Close to Zero Grad ---")
        print("PT = 0.0, GT = 0.0, T = 0.5")
        threshold = torch.tensor([0.5])
        gt = torch.tensor([0.0])
        pt = torch.tensor([0.0], requires_grad=True)
        
        inputs = (gt, pt, threshold)
        gc_test = torch.autograd.gradcheck(l_tp, inputs, eps=1e-2, atol=1e-2, raise_exception=True)
        jacob = torch.autograd.functional.jacobian(l_tp, inputs=inputs)

        print("Heaviside: {}".format(l_tp(gt, pt, threshold)))
        print("Jacobian: {}".format(jacob))

        res = l_tp(gt, pt, threshold)
        res.backward(torch.ones_like(pt))
        print("Gradient: {} \n".format(pt.grad.data))


    def test_grad_l_tp_pos(self):
        """ Testing TP 
        - Should be positive as we move towards the GT. 
        """
        print("--- TP - Positive Grad ---")
        print("PT = 0.8, GT = 1.0, T = 0.5")
        threshold = torch.tensor([0.5])
        gt = torch.tensor([1.0])
        pt = torch.tensor([0.8], requires_grad=True)

        inputs = (gt, pt, threshold)
        gc_test = torch.autograd.gradcheck(
            l_tp, inputs, eps=1e-2, atol=1e-2, raise_exception=True)
        jacob = torch.autograd.functional.jacobian(l_tp, inputs=inputs)

        print("Heaviside: {}".format(l_tp(gt, pt, threshold)))
        print("Jacobian: {}".format(jacob))

        res = l_tp(gt, pt, threshold)
        res.backward(torch.ones_like(pt))
        print("Gradient: {}\n".format(pt.grad.data))


    def test_grad_l_tp_neg(self):
        """ Testing TP
        - Should be negative as we move towards the GT of 0. 
        """
        print("--- TP - Negative Grad ---")
        print("PT = 0.6, GT = 0.0, T = 0.5")
        threshold = torch.tensor([0.5])
        gt = torch.tensor([0.0])
        pt = torch.tensor([0.6], requires_grad=True)

        inputs = (gt, pt, threshold)
        gc_test = torch.autograd.gradcheck(
            l_tp, inputs, eps=1e-2, atol=1e-2, raise_exception=True)
        jacob = torch.autograd.functional.jacobian(l_tp, inputs=inputs)

        print("Heaviside: {}".format(l_tp(gt, pt, threshold)))
        print("Jacobian: {}".format(jacob))

        res = l_tp(gt, pt, threshold)
        res.backward(torch.ones_like(pt))
        print("Gradient: {}\n".format(pt.grad.data))


    ############ FALSE POSITIVES ############
    def test_grad_l_fp_pos(self): 
        """ Testing FP 
        - Should be posiitve as we push towards being a FP.
        """
        print("--- FP - Positive Grad ---")
        print("PT = 0.9, GT = 0.0, T = 0.5")
        
        threshold = torch.tensor([0.5])
        gt = torch.tensor([0.0])
        pt = torch.tensor([0.9], requires_grad=True)

        inputs = (gt, pt, threshold)
        gc_test = torch.autograd.gradcheck(
            l_fp, inputs, eps=1e-2, atol=1e-2, raise_exception=True)
        jacob = torch.autograd.functional.jacobian(l_tp, inputs=inputs)

        print("Heaviside: {}".format(l_tp(gt, pt, threshold)))
        print("Jacobian: {}".format(jacob))

        res = l_fp(gt, pt, threshold)
        res.backward(torch.ones_like(pt))
        print("Gradient: {}\n".format(pt.grad.data))
    
    ########### FALSE NEGATIVES ############
    def test_grad_l_fn_neg(self):
        """ Testing FN - Appears negative when it's not. 
        - Downward pressure because it's trying to detect false negatives. 
        """
        print("--- FN - Negative Grad ---")
        print("PT = 0.3, GT = 1.0, T = 0.5")
        warnings.filterwarnings(action="ignore")
        threshold = torch.tensor([0.5])
        gt = torch.tensor([1.0])
        pt = torch.tensor([0.3], requires_grad=True)

        inputs = (gt, pt, threshold)
        gc_test = torch.autograd.gradcheck(
            l_fn, inputs, eps=1e-2, atol=1e-2, raise_exception=True)
        jacob = torch.autograd.functional.jacobian(l_tp, inputs=inputs)

        print("Heaviside: {}".format(l_tp(gt, pt, threshold)))
        print("Jacobian: {}".format(jacob))

        res = l_fn(gt, pt, threshold)
        res.backward(torch.ones_like(pt))
        print("Gradient: {}\n".format(pt.grad.data))


    def test_grad_l_fn_neg_1(self):
        """ Testing FN - Appears negative when it's not.
        - Upward pressure because it's trying to detect false negatives. 
        """
        print("--- FN - Negative Grad ---")
        print("PT = 0.7, GT = 1.0, T = 0.5")
        warnings.filterwarnings(action="ignore")
        threshold = torch.tensor([0.5])
        gt = torch.tensor([1.0])
        pt = torch.tensor([0.7], requires_grad=True)

        inputs = (gt, pt, threshold)
        gc_test = torch.autograd.gradcheck(
            l_fn, inputs, eps=1e-2, atol=1e-2, raise_exception=True)
        jacob = torch.autograd.functional.jacobian(l_tp, inputs=inputs)

        print("Heaviside: {}".format(l_tp(gt, pt, threshold)))
        print("Jacobian: {}".format(jacob))

        res = l_fn(gt, pt, threshold)
        res.backward(torch.ones_like(pt))
        print("Gradient: {}\n".format(pt.grad.data))

    def test_grad_l_fn_zero(self):
        """ Testing FN - Appears negative when it's not.
        """
        print("--- FN - Close to Zero Grad ---")
        print("PT = 0.0, GT = 1.0, T = 0.5")
        warnings.filterwarnings(action="ignore")
        threshold = torch.tensor([0.5])
        gt = torch.tensor([1.0])
        pt = torch.tensor([0.0], requires_grad=True)

        inputs = (gt, pt, threshold)
        gc_test = torch.autograd.gradcheck(
            l_fn, inputs, eps=1e-2, atol=1e-2, raise_exception=True)
        jacob = torch.autograd.functional.jacobian(l_tp, inputs=inputs)

        print("Heaviside: {}".format(l_tp(gt, pt, threshold)))
        print("Jacobian: {}".format(jacob))

        res = l_fn(gt, pt, threshold)
        res.backward(torch.ones_like(pt))
        print("Gradient: {}\n".format(pt.grad.data))


    ########### TRUE NEGATIVES ############
    def test_grad_l_tn_zero(self):
        """ Testing TN - Should be close 0 gradient. 
        - GT == PT, belongs in TN. 
        """
        print("--- TN - Close to Zero Grad ---")
        print("PT = 0.0, GT = 0.0, T = 0.5")
        warnings.filterwarnings(action="ignore")
        threshold = torch.tensor([0.5])
        gt = torch.tensor([0.0])
        pt = torch.tensor([0.0], requires_grad=True)

        inputs = (gt, pt, threshold)
        gc_test = torch.autograd.gradcheck(
            l_tn, inputs, eps=1e-2, atol=1e-2, raise_exception=True)
        jacob = torch.autograd.functional.jacobian(l_tp, inputs=inputs)

        print("Heaviside: {}".format(l_tp(gt, pt, threshold)))
        print("Jacobian: {}".format(jacob))

        res = l_tn(gt, pt, threshold)
        res.backward(torch.ones_like(pt))
        print("Gradient: {}\n".format(pt.grad.data))


    def test_grad_l_tn_neg_1(self):
        """ Testing TN - It is negative. 
        - Not a member of TN b/c of GT = 1, thus TN is always 0. 
        """
        print("--- TN - Negative Grad ---")
        print("PT = 0.7, GT = 1.0, T = 0.5")
        warnings.filterwarnings(action="ignore")
        gt = torch.tensor([1.0])
        pt = torch.tensor([0.7], requires_grad=True)
        threshold = torch.tensor([0.5])

        inputs = (gt, pt, threshold)
        gc_test = torch.autograd.gradcheck(
            l_tn, inputs, eps=1e-2, atol=1e-2, raise_exception=True)
        jacob = torch.autograd.functional.jacobian(l_tp, inputs=inputs)

        print("Heaviside: {}".format(l_tp(gt, pt, threshold)))
        print("Jacobian: {}".format(jacob))

        res = l_tn(gt, pt, threshold)
        res.backward(torch.ones_like(pt))
        print("Gradient: {}\n".format(pt.grad.data))


    def test_grad_l_tn_neg_2(self):
        """ Testing TN - It is Negative. 
        - PT should still be pushed to 0. 
        """
        print("--- TN - Negative Grad ---")
        print("PT = 0.3, GT = 0.0, T = 0.5")
        warnings.filterwarnings(action="ignore")
        threshold = torch.tensor([0.5])
        gt = torch.tensor([0.0])
        pt = torch.tensor([0.3], requires_grad=True)

        inputs = (gt, pt, threshold)

        gc_test = torch.autograd.gradcheck(
            l_tn, inputs, eps=1e-2, atol=1e-2, raise_exception=True)
        jacob = torch.autograd.functional.jacobian(l_tp, inputs=inputs)

        print("Heaviside: {}".format(l_tp(gt, pt, threshold)))
        print("Jacobian: {}".format(jacob))

        res = l_tn(gt, pt, threshold)
        res.backward(torch.ones_like(pt))
        print("Gradient: {}\n".format(pt.grad.data))
    


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
