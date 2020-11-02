#!/usr/bin/env python3
import numpy as np
import pandas as pd
import torch
import logging
from torch.autograd import Variable

from torch.autograd import gradcheck

EPS = 1e-7


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
        conditions = torch.where(cm1, torch.tensor([1.0]), torch.where(cm3, torch.tensor([3.0]), torch.tensor([2.0])))
        print('x', x)
        print('t', t)
        print('conditions', conditions)
        # print(f"m1 = {d}/{t}-{tt}/2")
        # print(f"m2 = (1-2*{d})/({tt}+EPS)")
        # print(f"m3 = {d}/(1-{t}-{tt}/2)")
    res = torch.where(cm1,m1*x,torch.where(cm3, m3*x + (1-d-m3*(t+tt/2)), m2*(x-t)+0.5))
    if debug:
        print('res', res)
    return res


def heaviside_agg(xs, thresholds, agg):
    # res = torch.zeros_like(xs)
    # for i in range(xs.shape[0]):
    #     res[i] = heaviside_approx(xs[i], thresholds[i])
    
    new_res = heaviside_approx(xs, thresholds)

    if agg == 'sum':
        return torch.sum(new_res, axis=0)

    return new_res


def l_tp(gt, pt, thresh, agg='sum'):
    # output closer to 1 if a true positive, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 1 -> (inverter = false)
    #  fn: (gt == 1 and pt == 0) -> closer to 0 -> (inverter = false)
    #  fp: (gt == 0 and pt == 1) -> closer to 0 -> (inverter = true)
    #  tn: (gt == 0 and pt == 0) -> closer to 0 -> (inverter = false)
    thresh = torch.where(thresh == 0.0, torch.tensor([0.01], device=thresh.device),
                         torch.where(thresh == 1.0, torch.tensor([0.99], device=thresh.device), thresh))
    gt_t = torch.reshape(torch.repeat_interleave(gt, thresh.shape[0]), (-1, thresh.shape[0])).to(thresh.device)
    pt_t = torch.reshape(torch.repeat_interleave(pt, thresh.shape[0]), (-1, thresh.shape[0])).to(thresh.device)
    condition = (gt_t == 0) & (pt_t >= thresh)
    xs = torch.where(condition, 1-pt_t, pt_t)
    # print("TP XS: {}".format(xs))
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
        gt, thresh.shape[0]), (-1, thresh.shape[0])).to(thresh.device)
    pt_t = torch.reshape(torch.repeat_interleave(
        pt, thresh.shape[0]), (-1, thresh.shape[0])).to(thresh.device)
    condition = (gt_t == 0) & (pt_t < thresh)

    # 1-pt_t might not actually work
    # want to make sure that in the first case, we should have pos
    # need to make sure that the graident is pushing it towards the right direction
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
    gt_t = torch.reshape(torch.repeat_interleave(gt, thresh.shape[0]), (-1, thresh.shape[0])).to(thresh.device)
    pt_t = torch.reshape(torch.repeat_interleave(pt, thresh.shape[0]), (-1, thresh.shape[0])).to(thresh.device)
    condition = (gt_t == 1) & (pt_t >= thresh)
    xs = torch.where(condition, 1-pt_t, pt_t)

    thresholds = torch.where(condition, 1-thresh, thresh)
    value = heaviside_agg(xs, thresholds, agg)

    return value


def l_tn(gt, pt, thresh, agg='sum'):
    # output closer to 1 if a true negative, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 0 -> (invert = true)
    #  fn: (gt == 1 and pt == 0) -> closer to 0 -> (invert = false)
    #  fp: (gt == 0 and pt == 1) -> closer to 0 -> (invert = true)
    #  tn: (gt == 0 and pt == 0) -> closer to 1 -> (invert = true)

    # thresh: tensor([0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000, 0.9000])
    thresh = torch.where(thresh == 0.0, torch.tensor([0.01], device=thresh.device),
                         torch.where(thresh == 1.0, torch.tensor([0.99], device=thresh.device), thresh))

    # GT_T: tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    # GT_T: tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1.]])
    gt_t = torch.reshape(torch.repeat_interleave(
        gt, thresh.shape[0]), (-1, thresh.shape[0])).to(thresh.device)

    # PT_T: tensor([[0.9921, 0.9921, 0.9921, 0.9921, 0.9921, 0.9921, 0.9921, 0.9921, 0.9921]],
    pt_t = torch.reshape(torch.repeat_interleave(
        pt, thresh.shape[0]), (-1, thresh.shape[0])).to(thresh.device)

    condition = (gt_t == 1) & (pt_t < thresh)

    # if it matches the threshold, keep the pt; otherwise, flip it.
    xs = torch.where(condition, pt_t, 1-pt_t)

    # print("XS: {}".format(xs.shape))

    # thresholds: tensor([[0.9000, 0.8000, 0.7000, 0.6000, 0.5000, 0.4000, 0.3000, 0.2000, 0.1000]])
    thresholds = torch.where(condition, thresh, 1-thresh)
    return heaviside_agg(xs, thresholds, agg)


def confusion(gt, pt, thresholds, agg='sum'):
    tp = l_tp(gt, pt, thresholds, agg)
    fn = l_fn(gt, pt, thresholds, agg)
    fp = l_fp(gt, pt, thresholds, agg)
    tn = l_tn(gt, pt, thresholds, agg)
    return tp, fn, fp, tn


def mean_f1_approx_loss_on(device, y_labels=None, y_preds=None, thresholds=torch.arange(0.1, 1, 0.1)):
    ''' Mean of Heaviside Approx F1 
    F1 across the classes is evenly weighted, hence Macro F1 

    Args: 
        y_labels: one-hot encoded label, i.e. 2 -> [0, 0, 1] 
        y_preds: softmaxed predictions
    '''
    thresholds = thresholds.to(device)
    def loss(y_labels, y_preds):
        classes = len(y_labels[0])
        mean_f1s = torch.zeros(classes, dtype=torch.float32).to(device)
        for i in range(classes):
            gt_list = torch.Tensor([x[i] for x in y_labels])
            pt_list = y_preds[:, i]

            thresholds = torch.arange(0.1, 1, 0.1).to(device)
            tp, fn, fp, tn = confusion(gt_list, pt_list, thresholds)
            precision = tp/(tp+fp+EPS)
            recall = tp/(tp+fn+EPS)
            temp_f1 = torch.mean(2 * (precision * recall) /
                                 (precision + recall + EPS))
            mean_f1s[i] = temp_f1

        loss = 1 - mean_f1s.mean()
        return loss
    return loss


def mean_accuracy_approx_loss_on(device, y_labels=None, y_preds=None, thresholds=torch.arange(0.1, 1, 0.1)):
    ''' Mean of Heaviside Approx Accuracy 
    Accuracy across the classes is evenly weighted 
    '''
    thresholds = thresholds.to(device)
    def loss(y_labels, y_preds):
        classes = len(y_labels[0])
        mean_accs = torch.zeros(classes, dtype=torch.float32).to(device)
        for i in range(classes):
            gt_list = torch.Tensor([x[i] for x in y_labels])
            pt_list = y_preds[:, i]

            thresholds = torch.arange(0.1, 1, 0.1)
            tp, fn, fp, tn = confusion(gt_list, pt_list, thresholds)
            mean_accs[i] = torch.mean((tp + tn) / (tp + tn + fp + fn))
        loss = 1 - mean_accs.mean()
        return loss
    return loss


def area(x, y):
    ''' area under curve via trapezoidal rule '''
    direction = 1
    # the following is equivalent to: dx = np.diff(x)
    dx = x[1:] - x[:-1]
    if torch.any(dx < 0):
        if torch.all(dx <= 0):
            direction = -1
        else:
            logging.warn(
                "x is neither increasing nor decreasing\nx: {}\ndx: {}.".format(x, dx))
            return 0
    return direction * torch.trapz(y, x)


def mean_auroc_approx_loss_on(device, y_labels=None, y_preds=None, linspacing=11):
    def loss(y_labels, y_preds):
        """Approximate auroc:
            - Linear interpolated Heaviside function
            - roc (11-point approximation)
            - integrate via trapezoidal rule under curve
        """
        classes = len(y_labels[0])
        thresholds = torch.linspace(0, 1, linspacing).to(device)
        areas = []
        # mean over all classes
        for i in range(classes):
            gt_list = torch.Tensor([x[i] for x in y_labels])
            pt_list = y_preds[:, i]

            tp, fn, fp, tn = confusion(gt_list, pt_list, thresholds)
            fpr = fp/(fp+tn+EPS)
            tpr = tp/(tp+fn+EPS)
            a = area(fpr, tpr)
            if a > 0:
                areas.append(a)
        loss = 1 - torch.stack(areas).mean()
        return loss
    return loss
