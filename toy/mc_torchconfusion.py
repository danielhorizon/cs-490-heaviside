#!/usr/bin/env python3
import torch
import logging

EPS = 1e-7

'''
- each class output is a probability [0, 1] range (softmax output from network)
- for each class, the value at the current class array element is taken, then computed. 
- the output is not independent? 
'''

# step functions and approximations


def indicator():
    ''' heaviside step function '''
    def f(x, t):
        return torch.where(x < t, 0, 1)
    return f


def sig(k=10):
    '''
        Kyurkchiev and Markov 2015
        Simple sigmoid
            - limits do not necessarily converge to the heaviside function
            - derivative can be 0
        = a/(1+e^(-k*(x-threshold)))
        for simplicity: a == 1
    '''
    def f(x, t):
        # shift to threshold
        x = x - t
        return 1/(1+torch.exp(-k*x))
    return f


def linear_approx(delta=0.2):
    ''' piecewise linear approximation of the Heaviside function
        x, t: pre-inverted (x, threshold) values in a tuple
        shape is 1 x num_thresholds
    '''
    d = delta

    def f(x, t):
        tt = torch.min(t, 1-t)
        cm1 = x < t - tt/2
        m1 = d/(t-tt/2)
        m2 = (1-2*d)/(tt+EPS)
        cm3 = x > t + tt/2
        m3 = d/(1-t-tt/2)
        res = torch.where(cm1, m1*x,
                          torch.where(cm3, m3*x + (1-d-m3*(t+tt/2)),
                                      m2*(x-t)+0.5))
        return res
    return f


def heaviside_sum(xs, thresholds, approx=None, gt_weight=None):
    ''' xs.shape: [batchsize, thresholds]
        thresholds.shape: [batchsize, thresholds]
        approx: linear_approx or approximation function to use
        '''
    a = approx(xs, thresholds).cuda()
    if gt_weight is not None:
        a = a * gt_weight.repeat(9, 1).transpose(0, 1)
    return torch.sum(a, axis=0)

# confusion matrix values


def l_tp(gt, pt, thresh, approx=None, gt_weight=None):
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
    condition = (pt_t > thresh)
    xs = torch.where(gt_t > 0, pt_t, torch.zeros_like(gt_t, dtype=pt_t.dtype))
    thresholds = torch.where(condition, thresh, 1-thresh)
    return heaviside_sum(xs, thresholds, approx, gt_weight=gt_weight)


def l_fn(gt, pt, thresh, approx=None, gt_weight=None):
    # output closer to 1 if a false negative, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 0 -> (inverter = true)
    #  fn: (gt == 1 and pt == 0) -> closer to 1 -> (inverter = true)
    #  fp: (gt == 0 and pt == 1) -> closer to 0 -> (inverter = true)
    #  tn: (gt == 0 and pt == 0) -> closer to 0 -> (inverter = false)
    thresh = torch.where(thresh == 0.0, torch.tensor([0.01], device=thresh.device),
                         torch.where(thresh == 1.0, torch.tensor([0.99], device=thresh.device), thresh))
    gt_t = torch.reshape(torch.repeat_interleave(
        gt, thresh.shape[0]), (-1, thresh.shape[0]))
    pt_t = torch.reshape(torch.repeat_interleave(
        pt, thresh.shape[0]), (-1, thresh.shape[0]))
    condition = (pt_t < thresh)
    xs = torch.where(gt_t > 0, 1-pt_t,
                     torch.zeros_like(gt_t, dtype=pt_t.dtype))
    thresholds = torch.where(condition, 1-thresh, thresh)
    return heaviside_sum(xs, thresholds, approx, gt_weight=gt_weight)


def l_fp(gt, pt, thresh, approx=None, gt_weight=None):
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
    condition = (pt_t < thresh)
    xs = torch.where(gt_t > 0, torch.zeros_like(gt_t, dtype=pt_t.dtype), pt_t)
    thresholds = torch.where(condition, thresh, 1-thresh)
    return heaviside_sum(xs, thresholds, approx, gt_weight=gt_weight)


def l_tn(gt, pt, thresh, approx=None, gt_weight=None):
    # output closer to 1 if a true negative, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 0 -> (invert = true)
    #  fn: (gt == 1 and pt == 0) -> closer to 0 -> (invert = false)
    #  fp: (gt == 0 and pt == 1) -> closer to 0 -> (invert = true)
    #  tn: (gt == 0 and pt == 0) -> closer to 1 -> (invert = true)

    # thresh: tensor([0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000, 0.9000])
    thresh = torch.where(thresh == 0.0, torch.tensor([0.01], device=thresh.device),
                         torch.where(thresh == 1.0, torch.tensor([0.99], device=thresh.device), thresh))

    # GT_T: tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0.]]) -or-
    # GT_T: tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1.]])
    gt_t = torch.reshape(torch.repeat_interleave(
        gt, thresh.shape[0]), (-1, thresh.shape[0]))

    # PT_T: tensor([[0.9921, 0.9921, 0.9921, 0.9921, 0.9921, 0.9921, 0.9921, 0.9921, 0.9921]],
    pt_t = torch.reshape(torch.repeat_interleave(
        pt, thresh.shape[0]), (-1, thresh.shape[0]))

    condition = (pt_t < thresh)
    xs = torch.where(gt_t > 0, torch.zeros_like(
        gt_t, dtype=pt_t.dtype), 1-pt_t)

    # if it matches the threshold, keep the pt; otherwise, flip it.
    thresholds = torch.where(condition, 1-thresh, thresh)
    return heaviside_sum(xs, thresholds, approx, gt_weight=gt_weight)


def confusion(gt, pt, thresholds, approx=None, class_weight=None):
    gt_weight = None
    if class_weight is not None:
        gt_weight = torch.where(gt == 0, class_weight[0], class_weight[1])
    tp = l_tp(gt, pt, thresholds, approx, gt_weight=gt_weight)
    fn = l_fn(gt, pt, thresholds, approx, gt_weight=gt_weight)
    fp = l_fp(gt, pt, thresholds, approx, gt_weight=gt_weight)
    tn = l_tn(gt, pt, thresholds, approx, gt_weight=gt_weight)
    return tp, fn, fp, tn


# binary-classification metrics
def bce(gt, pt):
    # (1/N) * ( p * ln(q) + (1-p) * ln(1-q) )
    return -1/gt.shape[0] * (gt * torch.log(pt) + (1-gt) * torch.log(1-pt)).nansum()


def fbeta(gt, pt, thresholds, approx=indicator(), beta=1):
    tp, fn, fp, tn = confusion(gt, pt, thresholds, approx)
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    return (1 + beta*beta) * (precision * recall) / (beta*beta * precision + recall)


def accuracy(gt, pt, thresholds, approx=indicator()):
    tp, fn, fp, tn = confusion(gt, pt, thresholds, approx)
    return (tp + tn) / (tp + fn + fp + tn)


def kl(p, q):
    kl = p * torch.log(p/q)
    kl[torch.isnan(kl)] = 0
    kl[torch.isinf(kl)] = 0
    # print(f"kl: {kl}")
    return kl


def mean_f1_approx_loss_on(device, y_labels=None, y_preds=None, thresholds=torch.arange(0.1, 1, 0.1)):
    ''' Mean of Heaviside Approx F1 
    F1 across the classes is evenly weighted, hence Macro F1 

    Args: 
        y_labels: one-hot encoded label, i.e. 2 -> [0, 0, 1] 
        y_preds: softmaxed predictions
    '''
    thresholds = thresholds.to(device)

    def loss(y_labels, y_preds):
        classes = len(y_labels[0])  # getting num of classes
        # we store f1 score for each class in this tensor
        mean_f1s = torch.zeros(classes, dtype=torch.float32).to(device)

        # looping over each class
        for i in range(classes):
            gt_list = torch.Tensor([x[i] for x in y_labels])
            pt_list = y_preds[:, i]  # pt list for the given class

            thresholds = thresholds.to(device)

            tp, fn, fp, tn = confusion(
                gt_list, pt_list, thresholds, approx=linear_approx())

            precision = tp/(tp+fp+EPS)
            recall = tp/(tp+fn+EPS)
            # WE CAN CHANGE THIS
            class_f1 = torch.mean(2 * (precision * recall) /
                                  (precision + recall + EPS))
            mean_f1s[i] = class_f1
        # we can also change the weighting here!
        loss = 1 - mean_f1s.mean()
        return loss, tp, fn, fp, tn
    return loss


def mean_ap_approx_loss_on(device, y_labels=None, y_preds=None, thresholds=torch.arange(0.1, 1, 0.1)):
    thresholds = thresholds.to(device)

    def loss(y_labels, y_preds):
        """
        Approximate F1:
            - Linear interpolated Heaviside function
            - Harmonic mean of precision and recall
            - Mean over a range of thresholds
        Args:
            y_labels: one-hot encoded label, i.e. 2 -> [0, 0, 1]
            y_preds: softmaxed predictions
        """
        classes = len(y_labels[0])
        mean_f1s = torch.zeros(classes, dtype=torch.float32).to(device)
        # thresholds = torch.arange(0.1, 1, 0.1).to(device)

        y_labels = y_labels.to(device)
        y_preds = y_preds.to(device)

        tp, fn, fp, tn = confusion(
            device, y_labels, y_preds, thresholds, approx=linear_approx())
        precision = tp/(tp+fp+EPS)
        recall = tp/(tp+fn+EPS)

        # taking mean of all elements in the precision tensor
        loss = 1 - precision.mean()
        return loss
    return loss


# UNUSED METHODS
def mean_accuracy_approx_loss_on(device, y_labels=None, y_preds=None, thresholds=torch.arange(0.1, 1, 0.1)):
    ''' Mean of Heaviside Approx Accuracy 
    Accuracy across the classes is evenly weighted 
    '''
    thresholds = thresholds.to(device)

    def loss(y_labels, y_preds):
        classes = len(y_labels[0])
        mean_accs = torch.zeros(classes, dtype=torch.float32).to(device)
        thresholds = torch.arange(0.1, 1, 0.1).to(device)
        for i in range(classes):
            gt_list = torch.Tensor([x[i] for x in y_labels])
            pt_list = y_preds[:, i]

            thresholds = torch.arange(0.1, 1, 0.1)
            tp, fn, fp, tn = confusion(gt_list, pt_list, thresholds)
            mean_accs[i] = torch.mean((tp + tn) / (tp + tn + fp + fn))
        loss = 1 - mean_accs.mean()
        return loss
    return loss

# for mean auroc.


def area(x, y):
    ''' area under curve via trapezoidal rule '''
    direction = 1
    # the following is equivalent to: dx = np.diff(x)
    dx = x[1:] - x[:-1]
    if torch.any(dx < 0):
        if torch.all(dx <= 0):
            direction = -1
        else:
            # when you compute area under the curve using trapezoidal approx,
            # assume that the whole area under the curve is going one direction
            # compute one trapezoidal rule

            # INSTEAD, compute under every part of the curve.
            # TODO(dlee): compute from every single point, and compute
            # the trapezoidal rule under every single one of these points.
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
