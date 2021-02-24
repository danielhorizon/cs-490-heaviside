import torch
import time
import numpy as np

EPS = 1e-7


def lin(m=1, b=0):
    ''' just a line
    '''
    def f(x, t):
        # t is unused
        return m*x+b
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


def heaviside_sum(xs, thresholds, approx=None):
    ''' xs.shape: [batchsize, thresholds]
        thresholds.shape: [batchsize, thresholds]
        approx: linear_approx or approximation function to use
        '''
    return torch.sum(approx(xs, thresholds).cuda(), axis=0)


def l_fn(device, gt, pt, thresh, approx=None, class_val=None):
    # output closer to 1 if a false negative, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 0 -> (inverter = true)
    #  fn: (gt == 1 and pt == 0) -> closer to 1 -> (inverter = true)
    #  fp: (gt == 0 and pt == 1) -> closer to 0 -> (inverter = true)
    #  tn: (gt == 0 and pt == 0) -> closer to 0 -> (inverter = false)
    thresh = torch.where(thresh == 0.0, torch.tensor([0.01], device=thresh.device),
                         torch.where(thresh == 1.0, torch.tensor([0.99], device=thresh.device), thresh))

    gt_t = torch.reshape(torch.repeat_interleave(
        gt, thresh.shape[0]), (-1, thresh.shape[0])).to(device)
    pt_t = torch.reshape(torch.repeat_interleave(
        pt, thresh.shape[0]), (-1, thresh.shape[0])).to(device)
    condition = (gt_t == 0) & (pt_t < thresh)
    xs = torch.where(condition, pt_t, 1-pt_t).to(device)
    thresholds = torch.where(condition, thresh, 1-thresh).to(device)
    return heaviside_sum(xs, thresholds, approx)


def l_fp(device, gt, pt, thresh, approx=None, class_val=None):
    # output closer to 1 if a false positive, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 0 -> (inverter = true)
    #  fn: (gt == 1 and pt == 0) -> closer to 0 -> (inverter = false)
    #  fp: (gt == 0 and pt == 1) -> closer to 1 -> (inverter = false)
    #  tn: (gt == 0 and pt == 0) -> closer to 0 -> (inverter = false)
    thresh = torch.where(thresh == 0.0, torch.tensor([0.01], device=thresh.device),
                         torch.where(thresh == 1.0, torch.tensor([0.99], device=thresh.device), thresh))

    gt_t = torch.reshape(torch.repeat_interleave(
        gt, thresh.shape[0]), (-1, thresh.shape[0])).to(device)

    pt_t = torch.reshape(torch.repeat_interleave(
        pt, thresh.shape[0]), (-1, thresh.shape[0])).to(device)

    condition = (gt_t == 1) & (pt_t >= thresh)
    xs = torch.where(condition, 1-pt_t, pt_t).to(device)
    thresholds = torch.where(condition, 1-thresh, thresh).to(device)
    return heaviside_sum(xs, thresholds, approx)


def l_tn(device, gt, pt, thresh, approx=None, class_val=None):
    # output closer to 1 if a true negative, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 0 -> (invert = true)
    #  fn: (gt == 1 and pt == 0) -> closer to 0 -> (invert = false)
    #  fp: (gt == 0 and pt == 1) -> closer to 0 -> (invert = true)
    #  tn: (gt == 0 and pt == 0) -> closer to 1 -> (invert = true)

    thresh = torch.where(thresh == 0.0, torch.tensor([0.01], device=thresh.device),
                         torch.where(thresh == 1.0, torch.tensor([0.99], device=thresh.device), thresh))
    gt_t = torch.reshape(torch.repeat_interleave(
        gt, thresh.shape[0]), (-1, thresh.shape[0])).to(device)
    pt_t = torch.reshape(torch.repeat_interleave(
        pt, thresh.shape[0]), (-1, thresh.shape[0])).to(device)
    condition = (gt_t == 1) & (pt_t < thresh)
    xs = torch.where(condition, pt_t, 1-pt_t).to(device)
    thresholds = torch.where(condition, thresh, 1-thresh).to(device)
    return heaviside_sum(xs, thresholds, approx)


def l_tp(device, gt, pt, thresh, approx=None, class_val=None):
    # output closer to 1 if a true positive, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 1 -> (inverter = false)
    #  fn: (gt == 1 and pt == 0) -> closer to 0 -> (inverter = false)
    #  fp: (gt == 0 and pt == 1) -> closer to 0 -> (inverter = true)
    #  tn: (gt == 0 and pt == 0) -> closer to 0 -> (inverter = false)
    thresh = torch.where(thresh == 0.0, torch.tensor([0.01], device=thresh.device),
                         torch.where(thresh == 1.0, torch.tensor([0.99], device=thresh.device), thresh))

    gt_t = torch.reshape(torch.repeat_interleave(
        gt, thresh.shape[0]), (-1, thresh.shape[0])).to(device)
    pt_t = torch.reshape(torch.repeat_interleave(
        pt, thresh.shape[0]), (-1, thresh.shape[0])).to(device)
    condition = (gt_t == 0) & (pt_t >= thresh)
    xs = torch.where(condition, 1-pt_t, pt_t).to(device)
    thresholds = torch.where(condition, 1-thresh, thresh).to(device)

    return heaviside_sum(xs, thresholds, approx)


def confusion(device, gt, pt, thresholds, approx=None, class_val=None):
    tp = l_tp(device, gt, pt, thresholds, approx, class_val)
    fn = l_fn(device, gt, pt, thresholds, approx, class_val)
    fp = l_fp(device, gt, pt, thresholds, approx, class_val)
    tn = l_tn(device, gt, pt, thresholds, approx, class_val)

    return tp, fn, fp, tn


def l_fn_adj(device, gt, pt, thresh, approx=None):
    # output closer to 1 if a false negative, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 0 -> (inverter = true)
    #  fn: (gt == 1 and pt == 0) -> closer to 1 -> (inverter = true)
    #  fp: (gt == 0 and pt == 1) -> closer to 0 -> (inverter = true)
    #  tn: (gt == 0 and pt == 0) -> closer to 0 -> (inverter = false)
    thresh = torch.where(thresh == 0.0, torch.tensor([0.01], device=thresh.device),
                         torch.where(thresh == 1.0, torch.tensor([0.99], device=thresh.device), thresh))

    n_classes = gt.shape[1]
    # batch size x num_class x num_thresholds
    gt_t = torch.reshape(
            torch.repeat_interleave(gt, thresh.shape[0]),
            (-1, n_classes, thresh.shape[0])
        ).to(device)
    pt_t = torch.reshape(
            torch.repeat_interleave(pt, thresh.shape[0]),
            (-1, n_classes, thresh.shape[0])
        ).to(device)

    condition = (gt_t == 0) & (pt_t < thresh)
    xs = torch.where(condition, pt_t, 1-pt_t).to(device)
    thresholds = torch.where(condition, thresh, 1-thresh).to(device)
    return heaviside_sum(xs, thresholds, approx)


def l_fp_adj(device, gt, pt, thresh, approx=None):
    '''
    gt -> batch size * num_classes  (8 x 10)
    pt -> batch size x num_classes  (8 x 10)

    gt_t -> (batch_size x num_classes) x num_thresh
    pt_t -> (batch_size x num_classes) x num_thresh

    after modifying thresh:
    thresh -> num_thresh x num_classes
    '''
    # output closer to 1 if a false positive, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 0 -> (inverter = true)
    #  fn: (gt == 1 and pt == 0) -> closer to 0 -> (inverter = false)
    #  fp: (gt == 0 and pt == 1) -> closer to 1 -> (inverter = false)
    #  tn: (gt == 0 and pt == 0) -> closer to 0 -> (inverter = false)
    thresh = torch.where(thresh == 0.0, torch.tensor([0.01], device=thresh.device),
                         torch.where(thresh == 1.0, torch.tensor([0.99], device=thresh.device), thresh))

    n_classes = gt.shape[1]
    # batch size x num_class x num_thresholds
    gt_t = torch.reshape(
            torch.repeat_interleave(gt, thresh.shape[0]),
            (-1, n_classes, thresh.shape[0])
        ).to(device)
    pt_t = torch.reshape(
            torch.repeat_interleave(pt, thresh.shape[0]),
            (-1, n_classes, thresh.shape[0])
        ).to(device)

    condition = (gt_t == 1) & (pt_t >= thresh)
    xs = torch.where(condition, 1-pt_t, pt_t).to(device)
    thresholds = torch.where(condition, 1-thresh, thresh).to(device)
    return heaviside_sum(xs, thresholds, approx)


def l_tn_adj(device, gt, pt, thresh, approx=None):
    # output closer to 1 if a true negative, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 0 -> (invert = true)
    #  fn: (gt == 1 and pt == 0) -> closer to 0 -> (invert = false)
    #  fp: (gt == 0 and pt == 1) -> closer to 0 -> (invert = true)
    #  tn: (gt == 0 and pt == 0) -> closer to 1 -> (invert = true)

    thresh = torch.where(thresh == 0.0, torch.tensor([0.01], device=thresh.device),
                         torch.where(thresh == 1.0, torch.tensor([0.99], device=thresh.device), thresh))

    n_classes = gt.shape[1]

    gt_t = torch.reshape(
            torch.repeat_interleave(gt, thresh.shape[0]),
            (-1, n_classes, thresh.shape[0])
        ).to(device)

    pt_t = torch.reshape(
            torch.repeat_interleave(pt, thresh.shape[0]),
            (-1, n_classes, thresh.shape[0])
        ).to(device)

    condition = (gt_t == 1) & (pt_t < thresh)
    xs = torch.where(condition, pt_t, 1-pt_t).to(device)
    thresholds = torch.where(condition, thresh, 1-thresh).to(device)
    return heaviside_sum(xs, thresholds, approx)

def l_tp_adj(device, gt, pt, thresh, approx=None):
    # replacing the thresholds
    # print("device :{}".format(device))
    thresh = torch.where(thresh == 0.0, torch.tensor([0.01], device=thresh.device),
                         torch.where(thresh == 1.0, torch.tensor([0.99], device=thresh.device), thresh)).to(device)

    n_classes = gt.shape[1]
    # batch size x num_class x num_thresholds
    gt_t = torch.reshape(
            torch.repeat_interleave(gt, thresh.shape[0]),
            (-1, n_classes, thresh.shape[0])
        ).to(device)
    pt_t = torch.reshape(
            torch.repeat_interleave(pt, thresh.shape[0]),
            (-1, n_classes, thresh.shape[0])
        ).to(device)

    condition = (gt_t == 0) & (pt_t >= thresh)
    xs = torch.where(condition, 1-pt_t, pt_t).to(device)
    thresholds = torch.where(condition, 1-thresh, thresh).to(device)
    return heaviside_sum(xs, thresholds, approx)


def confusion_adj(device, gt, pt, thresholds, approx=None):
    tp = l_tp_adj(device, gt, pt, thresholds, approx)
    fn = l_fn_adj(device, gt, pt, thresholds, approx)
    fp = l_fp_adj(device, gt, pt, thresholds, approx)
    tn = l_tn_adj(device, gt, pt, thresholds, approx)
    return tp, fn, fp, tn


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
        thresholds = torch.arange(0.1, 1, 0.1).to(device)

        y_labels = y_labels.to(device)
        y_preds = y_preds.to(device)

        tp, fn, fp, tn = confusion_adj(device, y_labels, y_preds, thresholds, approx)
        precision = tp/(tp+fp+EPS)
        recall = tp/(tp+fn+EPS)
        temp_f1 = torch.mean(2 * (precision * recall) /
                             (precision + recall + EPS), axis=1)
        loss = 1 - temp_f1.mean()

        return loss
    return loss



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
        thresholds = threshold.to(device)

        # New Method 
        tp, fn, fp, tn = confusion_adj(
            device, y_labels, y_preds, thresholds, approx)
        precision = tp/(tp+fp+EPS)
        recall = tp/(tp+fn+EPS)
        temp_f1 = torch.mean(2 * (precision * recall) /
                             (precision + recall + EPS), axis=1)
        loss = 1 - temp_f1.mean()

        # print("New Loss: {}".format(loss))

        # Legacy Method 
        # mean_f1s = torch.zeros(classes, dtype=torch.float32).to(device)
        # x = 0
        # for i in range(classes):
        #     gt_list = y_labels[:, i].to(device)
        #     pt_list = y_preds[:, i].to(device)

        #     tp, fn, fp, tn = confusion(
        #         device, gt_list, pt_list, thresholds, approx)

        #     precision = tp/(tp+fp+EPS)
        #     recall = tp/(tp+fn+EPS)
        #     temp_f1 = torch.mean(2 * (precision * recall) /
        #                          (precision + recall + EPS))
        #     mean_f1s[i] = temp_f1
        #     if x % 1000 == 0:
        #         print("Batch - TP: {:.3f}, FN: {:.3f}, FP: {:.3f}, TN:{:.3f}, PR: {:.3f}, RE: {:.3f}, F1: {:.3f}".format(
        #             tp.item(), fn.item(), tp.item(), tn.item(), precision.item(), recall.item(), temp_f1))
        #     x += 1

        # loss = 1 - mean_f1s.mean()
        # print("OG Loss: {}".format(loss))
        return loss
    return loss

'''
Current runs (1/9/2020)
- 
- python main-af1.py --gpu 1 /app/timeseries/imagenet/data
- python main-thresh.py --thresh 0.1 --gpu 2 /app/timeseries/imagenet/data
- python main-thresh.py --thresh 0.125 --gpu 3 /app/timeseries/imagenet/data

- python main-thresh.py --thresh 0.2 --gpu 0 /app/timeseries/imagenet/data
- python main-thresh.py --thresh 0.3 --gpu 3/app/timeseries/imagenet/data

can run 
- python main-thresh.py --thresh 0.4 --gpu 2 /app/timeseries/imagenet/data

1/12/20 

0: af1, ce  
1: 0.1, 0.4
2: 0.125, 0.5
3: 0.2, 0.3 


0.6, 0.7, 0.8, 0.9. 




python main.py --gpu 0 /app/timeseries/imagenet/data
python main-thresh.py --thresh 0.4 --gpu 1 /app/timeseries/imagenet/data
python main-thresh.py --thresh 0.5 --gpu 2 /app/timeseries/imagenet/data







Still have to run: 
- 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9. 
- Taking pre-trained Alexnet and fine-tuning it using this.



0.6, 0.7, 0.8, 0.9. 
python main-thresh.py --thresh 0.6 --gpu 0 /app/timeseries/data/imagenet-data
python main-thresh.py --thresh 0.7 --gpu 1 /app/timeseries/data/imagenet-data
python main-thresh.py --thresh 0.8 --gpu 2 /app/timeseries/data/imagenet-data
python main-thresh.py --thresh 0.9 --gpu 3 /app/timeseries/data/imagenet-data

'''
