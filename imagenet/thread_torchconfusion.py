import torch
# import flamegraph
from torch.multiprocessing import Pool, set_start_method
# import multiprocessing
from functools import partial

from threading import Thread 
from queue import Queue

EPS = 1e-7


def lin(m=1, b=0):
    ''' just a line
    '''
    def f(x, t):
        # t is unused
        return m*x+b
    return f


def sig(k=1):
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


def heaviside_sum(device, xs, thresholds):
    ''' xs.shape: [batchsize, thresholds]
        thresholds.shape: [batchsize, thresholds]
        approx: linear_approx or approximation function to use
        '''
    approx=linear_approx()
    return torch.sum(approx(xs, thresholds).to(device), axis=0)


def l_tp(device, gt, pt, thresh):
    # output closer to 1 if a true positive, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 1 -> (inverter = false)
    #  fn: (gt == 1 and pt == 0) -> closer to 0 -> (inverter = false)
    #  fp: (gt == 0 and pt == 1) -> closer to 0 -> (inverter = true)
    #  tn: (gt == 0 and pt == 0) -> closer to 0 -> (inverter = false)
    # thresh = torch.where(thresh == 0.0, torch.tensor([0.01], device=thresh.device),
    #                      torch.where(thresh == 1.0, torch.tensor([0.99], device=thresh.device), thresh))
    
    thresh = thresh.to(device)
    gt_t = torch.reshape(torch.repeat_interleave(gt, thresh.shape[0]), (-1, thresh.shape[0])).to(device)
    pt_t = torch.reshape(torch.repeat_interleave(pt, thresh.shape[0]), (-1, thresh.shape[0])).to(device)
    condition = (gt_t == 0) & (pt_t >= thresh)
    xs = torch.where(condition, 1-pt_t, pt_t).to(device)
    thresholds = torch.where(condition, 1-thresh, thresh).to(device)
    return heaviside_sum(device, xs, thresholds)


def l_fn(device, gt, pt, thresh):
    # output closer to 1 if a false negative, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 0 -> (inverter = true)
    #  fn: (gt == 1 and pt == 0) -> closer to 1 -> (inverter = true)
    #  fp: (gt == 0 and pt == 1) -> closer to 0 -> (inverter = true)
    #  tn: (gt == 0 and pt == 0) -> closer to 0 -> (inverter = false)
    # thresh = torch.where(thresh == 0.0, torch.tensor([0.01], device=thresh.device),
    #                      torch.where(thresh == 1.0, torch.tensor([0.99], device=thresh.device), thresh))
    
    thresh = thresh.to(device)
    gt_t = torch.reshape(torch.repeat_interleave(gt, thresh.shape[0]), (-1, thresh.shape[0])).to(device)
    pt_t = torch.reshape(torch.repeat_interleave(pt, thresh.shape[0]), (-1, thresh.shape[0])).to(device)
    condition = (gt_t == 0) & (pt_t < thresh)
    xs = torch.where(condition, pt_t, 1-pt_t).to(device)
    thresholds = torch.where(condition, thresh, 1-thresh).to(device)
    return heaviside_sum(device, xs, thresholds)


def l_fp(device, gt, pt, thresh):
    # output closer to 1 if a false positive, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 0 -> (inverter = true)
    #  fn: (gt == 1 and pt == 0) -> closer to 0 -> (inverter = false)
    #  fp: (gt == 0 and pt == 1) -> closer to 1 -> (inverter = false)
    #  tn: (gt == 0 and pt == 0) -> closer to 0 -> (inverter = false)
    # thresh = torch.where(thresh == 0.0, torch.tensor([0.01], device=thresh.device),
    #                      torch.where(thresh == 1.0, torch.tensor([0.99], device=thresh.device), thresh))
    
    thresh = thresh.to(device)
    gt_t = torch.reshape(torch.repeat_interleave(gt, thresh.shape[0]), (-1, thresh.shape[0])).to(device)
    pt_t = torch.reshape(torch.repeat_interleave(
        pt, thresh.shape[0]), (-1, thresh.shape[0])).to(device)
    condition = (gt_t == 1) & (pt_t >= thresh)
    xs = torch.where(condition, 1-pt_t, pt_t).to(device)
    thresholds = torch.where(condition, 1-thresh, thresh).to(device)
    return heaviside_sum(device, xs, thresholds)


# def l_tn(gt, pt, thresh, approx=None, device="cuda:0"):
def l_tn(device ,gt, pt, thresh):
    # output closer to 1 if a true negative, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 0 -> (invert = true)
    #  fn: (gt == 1 and pt == 0) -> closer to 0 -> (invert = false)
    #  fp: (gt == 0 and pt == 1) -> closer to 0 -> (invert = true)
    #  tn: (gt == 0 and pt == 0) -> closer to 1 -> (invert = true)
    # thresh = torch.where(thresh == 0.0, torch.tensor([0.01], device=thresh.device),
    #                      torch.where(thresh == 1.0, torch.tensor([0.99], device=thresh.device), thresh))
    
    thresh = thresh.to(device)
    gt_t = torch.reshape(torch.repeat_interleave(gt, thresh.shape[0]), (-1, thresh.shape[0])).to(device)
    pt_t = torch.reshape(torch.repeat_interleave(
        pt, thresh.shape[0]), (-1, thresh.shape[0])).to(device)
    condition = (gt_t == 1) & (pt_t < thresh)
    xs = torch.where(condition, pt_t, 1-pt_t).to(device)
    thresholds = torch.where(condition, thresh, 1-thresh).to(device)
    return heaviside_sum(device, xs, thresholds)


def confusion_helper(cs_metric, gt=None, pt=None, thresholds=None):
    if cs_metric == "tp": 
        device = "cuda:0"
        res = l_tp(device,gt, pt, thresholds)
    if cs_metric == "fp":
        device = "cuda:1"
        res = l_fp(device,gt, pt, thresholds)
    if cs_metric == "tn":
        device = "cuda:2"
        res = l_tn(device,gt, pt, thresholds)
    if cs_metric == "fn":
        device = "cuda:3"
        res = l_fn(device,gt, pt, thresholds)
    return res 

def confusion(device, gt, pt, thresholds):
    # print("device= {}".format(device))
    # tp = l_tp(gt, pt, thresholds, approx)
    # fn = l_fn(gt, pt, thresholds, approx)
    # fp = l_fp(gt, pt, thresholds, approx)
    # tn = l_tn(gt, pt, thresholds, approx)
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    data_list = ['tp', 'fp', 'tn', 'fn']
    pool = Pool(processes=4)

    prod_fxn = partial(confusion_helper, gt=gt.detach(),
                     pt=pt.detach(), thresholds=thresholds.detach())
    result_list = pool.map(prod_fxn, data_list)
    pool.close()
    pool.join()
    tp, fp, tn, fn = result_list[0], result_list[1], result_list[2], result_list[3]
    return tp, fn, fp, tn


# def confusion(device, gt, pt, thresholds, approx=None):
#     # print("device= {}".format(device))
#     tp = l_tp(gt, pt, thresholds, approx)
#     fn = l_fn(gt, pt, thresholds, approx)
#     fp = l_fp(gt, pt, thresholds, approx)
#     tn = l_tn(gt, pt, thresholds, approx)
#     return tp, fn, fp, tn


def thresh_mean_f1_approx_loss_on(device, threshold, y_labels=None, y_preds=None, approx=linear_approx()):
    ''' Mean of Heaviside Approx F1 
    F1 across the classes is evenly weighted, hence Macro F1 

    Args: 
        y_labels: one-hot encoded label, i.e. 2 -> [0, 0, 1] 
        y_preds: softmaxed predictions
    '''

    def loss(y_labels, y_preds):
        y_labels = y_labels.to(device)
        y_preds = y_preds.to(device)

        classes = len(y_labels[0])
        mean_f1s = torch.zeros(classes, dtype=torch.float32).to(device)
        x = 0
        for i in range(classes):
            gt_list = y_labels[:, i].to(device)
            pt_list = y_preds[:, i].to(device)
            thresholds = threshold.to(device)

            tp, fn, fp, tn = confusion(
                device, gt_list, pt_list, thresholds)
            
            tp=tp.to(device)
            fn=fn.to(device)
            fp=fp.to(device)
            tn=tn.to(device)

            precision = tp/(tp+fp+EPS)
            recall = tp/(tp+fn+EPS)
            temp_f1 = torch.mean(2 * (precision * recall) /
                                 (precision + recall + EPS))
            mean_f1s[i] = temp_f1
            if x % 1000 == 0:
                print("Batch - TP: {:.3f}, FN: {:.3f}, FP: {:.3f}, TN:{:.3f}, PR: {:.3f}, RE: {:.3f}, F1: {:.3f}".format(
                    tp.item(), fn.item(), tp.item(), tn.item(), precision.item(), recall.item(), temp_f1))
            x += 1

        loss = 1 - mean_f1s.mean()
        return loss
    return loss


# def thresh_mean_f1_approx_loss_on(device, threshold, y_labels=None, y_preds=None, approx=linear_approx()):
#     ''' Mean of Heaviside Approx F1 
#     F1 across the classes is evenly weighted, hence Macro F1 

#     Args: 
#         y_labels: one-hot encoded label, i.e. 2 -> [0, 0, 1] 
#         y_preds: softmaxed predictions
#     '''

#     def loss(y_labels, y_preds):
#         y_labels = y_labels.to(device)
#         y_preds = y_preds.to(device)

#         classes = len(y_labels[0])
#         mean_f1s = torch.zeros(classes, dtype=torch.float32).to(device)
#         x = 0 
#         for i in range(classes):
#             gt_list = y_labels[:, i].to(device)
#             pt_list = y_preds[:, i].to(device)
#             thresholds = threshold.to(device)

#             tp, fn, fp, tn = confusion(device, gt_list, pt_list, thresholds, approx)

#             precision = tp/(tp+fp+EPS)
#             recall = tp/(tp+fn+EPS)
#             temp_f1 = torch.mean(2 * (precision * recall) /
#                                  (precision + recall + EPS))
#             mean_f1s[i] = temp_f1
#             if x % 1000 == 0:
#                 print("Batch - TP: {:.3f}, FN: {:.3f}, FP: {:.3f}, TN:{:.3f}, PR: {:.3f}, RE: {:.3f}, F1: {:.3f}".format(
#                     tp.item(), fn.item(), tp.item(), tn.item(), precision.item(), recall.item(), temp_f1))
#             x += 1 
            
#         loss = 1 - mean_f1s.mean()
#         return loss
#     return loss


def mean_f1_approx_loss_on(device, y_labels=None, y_preds=None, thresholds=torch.arange(0.1, 1, 0.1), approx=linear_approx()):
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

        for i in range(classes):
            gt_list = torch.Tensor([x[i] for x in y_labels]).to(device)
            pt_list = y_preds[:, i].to(device)

            thresholds = torch.arange(0.1, 1, 0.1).to(device)
            tp, fn, fp, tn = confusion(device, gt_list, pt_list, thresholds, approx)
            precision = tp/(tp+fp+EPS)
            recall = tp/(tp+fn+EPS)
            temp_f1 = torch.mean(2 * (precision * recall) /
                                 (precision + recall + EPS))
            mean_f1s[i] = temp_f1

        loss = 1 - mean_f1s.mean()
        return loss
    return loss

'''
Ports: 7013, 9998, 3334, 4443, 7776

0: 
python main-af1.py --gpu 0 /app/timeseries/imagenet/data
python main-thresh.py --thresh 0.2 --gpu 0 /app/timeseries/imagenet/data

1: 
python main-thresh.py --thresh 0.3 --gpu 1 /app/timeseries/imagenet/data
python main-thresh.py --thresh 0.4 --gpu 1 /app/timeseries/imagenet/data

2: 
python main-thresh.py --thresh 0.1 --gpu 2 /app/timeseries/imagenet/data
python main-thresh.py --thresh 0.5 --gpu 2 /app/timeseries/imagenet/data

3: 
python main-thresh.py --thresh 0.125 --gpu 3 /app/timeseries/imagenet/data
python main-thresh.py --thresh 0.6 --gpu 3 /app/timeseries/imagenet/data

LEFT TO RUN:
python main-thresh.py --thresh 0.7 --gpu 0 /app/timeseries/imagenet/data
python main-thresh.py --thresh 0.8 --gpu 1 /app/timeseries/imagenet/data
python main-thresh.py --thresh 0.9 --gpu 2 /app/timeseries/imagenet/data
python main-thresh.py --thresh 0.05 --gpu 3 /app/timeseries/imagenet/data


FlameGraph your pytorch impelemntation 
'''
