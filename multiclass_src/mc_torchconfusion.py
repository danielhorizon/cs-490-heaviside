#!/usr/bin/env python3
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

EPS = 1e-7


def heaviside_approx(x, t, delta=0.1, debug=False):
    ''' piecewise linear approximation of the Heaviside function
        x, t: pre-inverted (x, threshold) values in a tuple
        shape is 1 x num_thresholds
    '''
    d = delta
    tt = torch.min(t, 1-t)
    cm1 = x < t - tt/2
    m1 = d/(t-tt/2)
    m2 = (1-2*d)/(tt+EPS)
    cm3 = x > t + tt/2
    m3 = d/(1-t-tt/2)
    if debug:
        conditions = torch.where(cm1, torch.tensor([1.0]),
                                 torch.where(cm3, torch.tensor([3.0]),
                                             torch.tensor([2.0])))
        print('x', x)
        print('t', t)
        print('conditions', conditions)
        # print(f"m1 = {d}/{t}-{tt}/2")
        # print(f"m2 = (1-2*{d})/({tt}+EPS)")
        # print(f"m3 = {d}/(1-{t}-{tt}/2)")
    res = torch.where(
            cm1, 
            m1*x,
            torch.where(cm3, m3*x + (1-d-m3*(t+tt/2)), m2*(x-t)+0.5)
        )
    if debug:
        print('res', res)
    return res


def heaviside_agg(xs, thresholds, agg):
    res = torch.zeros_like(xs)

    for i in range(xs.shape[0]):
        res[i] = heaviside_approx(xs[i], thresholds[i])

    if agg == 'sum':
        return torch.sum(res, axis=0)
    return res


def l_tp(gt, pt, thresh, agg='sum'):
    # output closer to 1 if a true positive, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 1 -> (inverter = false)
    #  fn: (gt == 1 and pt == 0) -> closer to 0 -> (inverter = false)
    #  fp: (gt == 0 and pt == 1) -> closer to 0 -> (inverter = true)
    #  tn: (gt == 0 and pt == 0) -> closer to 0 -> (inverter = false)
    thresh = torch.where(thresh == 0.0, torch.tensor([0.01], device=thresh.device),
                         torch.where(thresh == 1.0, torch.tensor([0.99], device=thresh.device), thresh))
    gt_t = torch.reshape(torch.repeat_interleave(
        gt, thresh.shape[0]), (-1, thresh.shape[0]))
    pt_t = torch.reshape(torch.repeat_interleave(
        pt, thresh.shape[0]), (-1, thresh.shape[0]))
    condition = (gt_t == 0) & (pt_t >= thresh)
    xs = torch.where(condition, 1-pt_t, pt_t)
    thresholds = torch.where(condition, 1-thresh, thresh)
    return heaviside_agg(xs, thresholds, agg)


def l_fn(gt, pt, thresh, agg='sum'):
    # output closer to 1 if a false negative, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 0 -> (inverter = true)
    #  fn: (gt == 1 and pt == 0) -> closer to 1 -> (inverter = true)
    #  fp: (gt == 0 and pt == 1) -> closer to 0 -> (inverter = true)
    #  tn: (gt == 0 and pt == 0) -> closer to 0 -> (inverter = false)

    # filling in the places where it's 0 or 1.
    thresh = torch.where(
        thresh == 0.0,
        torch.tensor([0.01], device=thresh.device),
        torch.where(
            thresh == 1.0,
            torch.tensor([0.99], device=thresh.device), thresh)
    )

    gt_t = torch.reshape(torch.repeat_interleave(
        gt, thresh.shape[0]), (-1, thresh.shape[0]))
    pt_t = torch.reshape(torch.repeat_interleave(
        pt, thresh.shape[0]), (-1, thresh.shape[0]))
    condition = (gt_t == 0) & (pt_t < thresh)
    xs = torch.where(condition, pt_t, 1-pt_t)
    thresholds = torch.where(condition, thresh, 1-thresh)
    return heaviside_agg(xs, thresholds, agg)


def l_fp(gt, pt, thresh, agg='sum'):
    # output closer to 1 if a false positive, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 0 -> (inverter = true)
    #  fn: (gt == 1 and pt == 0) -> closer to 0 -> (inverter = false)
    #  fp: (gt == 0 and pt == 1) -> closer to 1 -> (inverter = false)
    #  tn: (gt == 0 and pt == 0) -> closer to 0 -> (inverter = false)
    thresh = torch.where(thresh == 0.0, torch.tensor([0.01], device=thresh.device),
                         torch.where(thresh == 1.0, torch.tensor([0.99], device=thresh.device), thresh))
    gt_t = torch.reshape(torch.repeat_interleave(
        gt, thresh.shape[0]), (-1, thresh.shape[0]))
    pt_t = torch.reshape(torch.repeat_interleave(
        pt, thresh.shape[0]), (-1, thresh.shape[0]))
    condition = (gt_t == 1) & (pt_t >= thresh)
    xs = torch.where(condition, 1-pt_t, pt_t)
    thresholds = torch.where(condition, 1-thresh, thresh)
    return heaviside_agg(xs, thresholds, agg)


def l_tn(gt, pt, thresh, agg='sum'):
    # output closer to 1 if a true negative, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 0 -> (invert = true)
    #  fn: (gt == 1 and pt == 0) -> closer to 0 -> (invert = false)
    #  fp: (gt == 0 and pt == 1) -> closer to 0 -> (invert = true)
    #  tn: (gt == 0 and pt == 0) -> closer to 1 -> (invert = true)
    thresh = torch.where(thresh == 0.0, torch.tensor([0.01], device=thresh.device),
                         torch.where(thresh == 1.0, torch.tensor([0.99], device=thresh.device), thresh))
    gt_t = torch.reshape(torch.repeat_interleave(
        gt, thresh.shape[0]), (-1, thresh.shape[0]))
    pt_t = torch.reshape(torch.repeat_interleave(
        pt, thresh.shape[0]), (-1, thresh.shape[0]))
    condition = (gt_t == 1) & (pt_t < thresh)
    xs = torch.where(condition, pt_t, 1-pt_t)
    thresholds = torch.where(condition, thresh, 1-thresh)
    return heaviside_agg(xs, thresholds, agg)


def confusion(gt, pt, thresholds, agg='sum'):
    # print("gt: {}".format(gt))
    # print("pt: {}".format(pt))
    tp = l_tp(gt, pt, thresholds, agg)
    fn = l_fn(gt, pt, thresholds, agg)
    fp = l_fp(gt, pt, thresholds, agg)
    tn = l_tn(gt, pt, thresholds, agg)
    return tp, fn, fp, tn


def mean_f1_approx_loss_on(y_labels=None, y_preds=None, thresholds=torch.arange(0.1, 1, 0.1)):
    # number of classes should be length of each element
    def loss(y_labels, y_preds): 
        classes = len(y_labels[0])
        mean_f1s = torch.zeros(classes, dtype=torch.float32)
        for i in range(classes):
            gt_list = torch.Tensor([x[i] for x in y_labels])
            # pt needs to be differentiable 
            # pt_list = Variable(y_preds[:, i], requires_grad=True)
            pt_list = y_preds[:, i]

            thresholds = torch.arange(0.1, 1, 0.1)
            # returns the number of tp, fn, fp, and tn.
            tp, fn, fp, _ = confusion(gt_list, pt_list, thresholds)
            precision = tp/(tp+fp+EPS)
            recall = tp/(tp+fn+EPS)
            temp_f1 = torch.mean(2 * (precision * recall) /
                                 (precision + recall + EPS))
            mean_f1s[i] = temp_f1
        loss = 1- mean_f1s.mean() 
        print("mean F1: {}".format(mean_f1s))
        print("loss: {}".format(loss))
        loss = Variable(loss, requires_grad=True)
        
        print(loss.grad_fn)

        return loss
    return loss


def test(): 
    y_label = [1, 0, 0]
    y_pred = [0.5, 0.3, 0.2]

    print(mean_f1_approx_loss_on(y_labels=y_label,
                                 y_preds=y_pred, thresholds=torch.arange(0.1, 1, 0.1)))

if __name__ == "__main__":
    test()
'''
- soft set membership function - will let you backpropagate through the entire thing. this comes from  
having a probability to correspond that the sample is part of the set or not. 
- when the metrics are computed, how do we get from the softmax to the label? 

- each class output is a probability [0, 1] range. 
- softmax, and then for each class, take the value at the current class array element 
- take value from each part of the column, etc. 

the network is predicting all of them in parallel, so it's predicting relative probabilities 
the structure of the network s.t. the output is not independent. 


'''
