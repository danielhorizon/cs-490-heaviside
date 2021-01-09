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
    print("Old Xs Shape: {}".format(xs.shape))
    return torch.sum(approx(xs, thresholds).cuda(), axis=0)



def l_tp(device, gt, pt, thresh, approx=None):
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
    # print("Old Xs shape:{}".format(xs.shape))
    # print(xs)
    thresholds = torch.where(condition, 1-thresh, thresh).to(device)
    return heaviside_sum(xs, thresholds, approx)


def l_fn(device, gt, pt, thresh, approx=None):
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
    # print("OG XS:{}".format(xs))
    thresholds = torch.where(condition, thresh, 1-thresh).to(device)
    return heaviside_sum(xs, thresholds, approx)


def l_fp(device, gt, pt, thresh, approx=None):
    # output closer to 1 if a false positive, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 0 -> (inverter = true)
    #  fn: (gt == 1 and pt == 0) -> closer to 0 -> (inverter = false)
    #  fp: (gt == 0 and pt == 1) -> closer to 1 -> (inverter = false)
    #  tn: (gt == 0 and pt == 0) -> closer to 0 -> (inverter = false)
    # thresh = torch.where(thresh == 0.0, torch.tensor([0.01], device=thresh.device),
    #                      torch.where(thresh == 1.0, torch.tensor([0.99], device=thresh.device), thresh))
    print("--- old pt: {}".format(pt))
    gt_t = torch.reshape(torch.repeat_interleave(
        gt, thresh.shape[0]), (-1, thresh.shape[0])).to(device)
    
    pt_t = torch.reshape(torch.repeat_interleave(
        pt, thresh.shape[0]), (-1, thresh.shape[0])).to(device)
    print("--- old pt_t: {}".format(pt_t))
    condition = (gt_t == 1) & (pt_t >= thresh)
    xs = torch.where(condition, 1-pt_t, pt_t).to(device)
    thresholds = torch.where(condition, 1-thresh, thresh).to(device)
    return heaviside_sum(xs, thresholds, approx)


def l_tn(device, gt, pt, thresh, approx=None):
    # output closer to 1 if a true negative, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 0 -> (invert = true)
    #  fn: (gt == 1 and pt == 0) -> closer to 0 -> (invert = false)
    #  fp: (gt == 0 and pt == 1) -> closer to 0 -> (invert = true)
    #  tn: (gt == 0 and pt == 0) -> closer to 1 -> (invert = true)
    
    thresh = torch.where(thresh == 0.0, torch.tensor([0.01], device=thresh.device),
                         torch.where(thresh == 1.0, torch.tensor([0.99], device=thresh.device), thresh))
    # print(thresh.shape) 
    # thresh = length = 9 
    # gt shape --> batch size 
    # pt shape --> batch size 
    gt_t = torch.reshape(torch.repeat_interleave(
        gt, thresh.shape[0]), (-1, thresh.shape[0])).to(device)
    # print(gt_t.shape)
    pt_t = torch.reshape(torch.repeat_interleave(
        pt, thresh.shape[0]), (-1, thresh.shape[0])).to(device)
    # print(pt_t.shape)
    condition = (gt_t == 1) & (pt_t < thresh)
    xs = torch.where(condition, pt_t, 1-pt_t).to(device)
    
    thresholds = torch.where(condition, thresh, 1-thresh).to(device)
    return heaviside_sum(xs, thresholds, approx)


def confusion(device, gt, pt, thresholds, approx=None):
    tp = l_tp(device, gt, pt, thresholds, approx)
    fn = l_fn(device, gt, pt, thresholds, approx)
    fp = l_fp(device, gt, pt, thresholds, approx)
    tn = l_tn(device, gt, pt, thresholds, approx)
    print("old TP: {}".format(tp))
    return tp, fn, fp, tn


def heaviside_sum_adj(xs, thresholds, approx=None):
    ''' xs.shape: [batchsize, thresholds]
        thresholds.shape: [batchsize, thresholds]
        approx: linear_approx or approximation function to use
        '''
    # axis = 0 was summing over all rows (i.e. for each column)
    print("New Xs shape: {}".format(xs.shape))
    return torch.sum(approx(xs, thresholds).cuda(), axis=0)

def l_tp_adj(device, gt, pt, thresh, approx=None):
    # output closer to 1 if a true positive, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 1 -> (inverter = false)
    #  fn: (gt == 1 and pt == 0) -> closer to 0 -> (inverter = false)
    #  fp: (gt == 0 and pt == 1) -> closer to 0 -> (inverter = true)
    #  tn: (gt == 0 and pt == 0) -> closer to 0 -> (inverter = false)
    thresh = torch.where(thresh == 0.0, torch.tensor([0.01], device=thresh.device),
                         torch.where(thresh == 1.0, torch.tensor([0.99], device=thresh.device), thresh))
    n_classes = gt.shape[1]
    gt_t = torch.reshape(
        torch.repeat_interleave(gt, thresh.shape[0]),
        (-1, thresh.shape[0], n_classes)).to(device)

    pt_t = torch.reshape(
        torch.repeat_interleave(
            pt, thresh.shape[0]
        ),
        (-1, thresh.shape[0], n_classes),
    ).to(device)
    # print(pt_t.shape)

    thresh = torch.reshape(
        torch.repeat_interleave(thresh, n_classes),
        (thresh.shape[0], n_classes)
    )

    # gt_t = torch.reshape(torch.repeat_interleave(
    #     gt, thresh.shape[0]), (-1, thresh.shape[0])).to(device)
    # pt_t = torch.reshape(torch.repeat_interleave(
    #     pt, thresh.shape[0]), (-1, thresh.shape[0])).to(device)
    condition = (gt_t == 0) & (pt_t >= thresh)
    xs = torch.where(condition, 1-pt_t, pt_t).to(device)
    thresholds = torch.where(condition, 1-thresh, thresh).to(device)
    return heaviside_sum_adj(xs, thresholds, approx)


def l_fn_adj(device, gt, pt, thresh, approx=None):
    # output closer to 1 if a false negative, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 0 -> (inverter = true)
    #  fn: (gt == 1 and pt == 0) -> closer to 1 -> (inverter = true)
    #  fp: (gt == 0 and pt == 1) -> closer to 0 -> (inverter = true)
    #  tn: (gt == 0 and pt == 0) -> closer to 0 -> (inverter = false)
    thresh = torch.where(thresh == 0.0, torch.tensor([0.01], device=thresh.device),
                         torch.where(thresh == 1.0, torch.tensor([0.99], device=thresh.device), thresh))

    # gt_t = torch.reshape(torch.repeat_interleave(
    #     gt, thresh.shape[0]), (-1, thresh.shape[0])).to(device)
    # pt_t = torch.reshape(torch.repeat_interleave(
    #     pt, thresh.shape[0]), (-1, thresh.shape[0])).to(device)
    n_classes = gt.shape[1]
    gt_t = torch.reshape(
        torch.repeat_interleave(gt, thresh.shape[0]),
        (-1, thresh.shape[0], n_classes)).to(device)

    pt_t = torch.reshape(
        torch.repeat_interleave(
            pt, thresh.shape[0]
        ),
        (-1, thresh.shape[0], n_classes),
    ).to(device)
    # print(pt_t.shape)

    thresh = torch.reshape(
        torch.repeat_interleave(thresh, n_classes),
        (thresh.shape[0], n_classes)
    )
    condition = (gt_t == 0) & (pt_t < thresh)
    xs = torch.where(condition, pt_t, 1-pt_t).to(device)
    # print("NEW XS: {}".format(xs[:, 0, :].shape))
    # print("VALUE OF 0th' class new XS:{}".format(xs[:,:,0]))
    thresholds = torch.where(condition, thresh, 1-thresh).to(device)
    return heaviside_sum_adj(xs, thresholds, approx)


def l_fp_adj(device, gt, pt, thresh, approx=None):
    # output closer to 1 if a false positive, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 0 -> (inverter = true)
    #  fn: (gt == 1 and pt == 0) -> closer to 0 -> (inverter = false)
    #  fp: (gt == 0 and pt == 1) -> closer to 1 -> (inverter = false)
    #  tn: (gt == 0 and pt == 0) -> closer to 0 -> (inverter = false)
    thresh = torch.where(thresh == 0.0, torch.tensor([0.01], device=thresh.device),
                         torch.where(thresh == 1.0, torch.tensor([0.99], device=thresh.device), thresh))

    # gt_t = torch.reshape(torch.repeat_interleave(
    #     gt, thresh.shape[0]), (-1, thresh.shape[0])).to(device)

    # pt_t = torch.reshape(torch.repeat_interleave(
    #     pt, thresh.shape[0]), (-1, thresh.shape[0])).to(device)
    n_classes = gt.shape[1]

    
        
        # num thresholds x num classes
    print("--thresh shape: {}".format(thresh.shape)) 

    # print("GT SHAPE: {}".format(gt.shape)) 
    print("--- new pt: {}".format(pt[:, 0]))

    '''
    gt -> batch size * num_classes  
    pt -> batch size x num_classes 
    
    gt_t -> (batch_size x num_classes) x num_thresh
    pt_t -> (batch_size x num_classes) x num_thresh

    after modifying thresh: 
    thresh -> num_thresh x num_classes
    '''
    
    gt_t = torch.reshape(
        torch.repeat_interleave(gt, thresh.shape[0]),
        (-1, thresh.shape[0], n_classes)
        ).to(device)
    
    print(torch.repeat_interleave(pt, thresh.shape[0]))
    pt_t = torch.reshape(
        torch.repeat_interleave(pt, thresh.shape[0]), 
        (thresh.shape[0], n_classes, -1,),
        ).to(device)

    print("ADJUSTED: {}".format( pt_t[:, :, 0]))
    

    thresh = torch.reshape(
        torch.repeat_interleave(thresh, n_classes),
        (thresh.shape[0], n_classes))
    print("Threshold shape: {}".format(thresh.shape))
    
    condition = (gt_t == 1) & (pt_t >= thresh)
    xs = torch.where(condition, 1-pt_t, pt_t).to(device)
    print("Adj Xs shape:{}".format(xs.shape))
    # print(xs)
    thresholds = torch.where(condition, 1-thresh, thresh).to(device)
    return heaviside_sum_adj(xs, thresholds, approx)


def l_tn_adj(device, gt, pt, thresh, approx=None):
    # output closer to 1 if a true negative, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 0 -> (invert = true)
    #  fn: (gt == 1 and pt == 0) -> closer to 0 -> (invert = false)
    #  fp: (gt == 0 and pt == 1) -> closer to 0 -> (invert = true)
    #  tn: (gt == 0 and pt == 0) -> closer to 1 -> (invert = true)

    thresh = torch.where(thresh == 0.0, torch.tensor([0.01], device=thresh.device),
                         torch.where(thresh == 1.0, torch.tensor([0.99], device=thresh.device), thresh))
    
    n_classes = gt.shape[1]
    # thresh = length = 9
    
    # gt shape --> batch size
    # pt shape --> batch size
    # print("Gt shape: {}".format(gt.shape))
    
    gt_t = torch.reshape(
        torch.repeat_interleave(gt, thresh.shape[0]), 
        (-1, thresh.shape[0], n_classes)).to(device)
    
    pt_t = torch.reshape(
        torch.repeat_interleave(
            pt, thresh.shape[0]
        ), 
        (-1, thresh.shape[0], n_classes), 
        ).to(device)
    # print(pt_t.shape)
    
    thresh = torch.reshape(
        torch.repeat_interleave(thresh, n_classes),
        (thresh.shape[0], n_classes)
    )
    condition = (gt_t == 1) & (pt_t < thresh)
    xs = torch.where(condition, pt_t, 1-pt_t).to(device)
    # print("XS shape:{}".format(xs.shape))
    thresholds = torch.where(condition, thresh, 1-thresh).to(device)
    x = heaviside_sum(xs, thresholds, approx)
    print("L_TN shape: {}".format(x.shape))
    # print(x)
    return x



def confusion_adj(device, gt, pt, thresholds, approx=None):
    tp = l_tp_adj(device, gt, pt, thresholds, approx)
    fn = l_fn_adj(device, gt, pt, thresholds, approx)
    fp = l_fp_adj(device, gt, pt, thresholds, approx)
    tn = l_tn_adj(device, gt, pt, thresholds, approx)
    print("new TP: {}".format(tp[:,0]))
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

        y_labels = y_labels.to(device)
        y_preds = y_preds.to(device)
        # thresholds = torch.arange(0.1, 1, 0.1).to(device)

        # print("classes: {}".format(classes))
        # print("y labels shape: {}".format(y_labels.shape))
        # print("y preds shape: {}".format(y_preds.shape))
        
        
        # print("tn :{}".format(tn))
        # print("tn shape:{}".format(tn.shape)) # 9 x 1000
        tp, fn, fp, tn = confusion_adj(
            device, y_labels, y_preds, thresholds, approx)
        precision = tp/(tp+fp+EPS)
        recall = tp/(tp+fn+EPS)
        temp_f1 = torch.mean(2 * (precision * recall) /  (precision + recall + EPS), axis=0)
        # print(precision.shape)
        # print(recall.shape)
        # computing it column wise 
        
        # print(temp_f1.shape)
        # print(temp_f1)
        # for i in range(classes):
        #     gt_list = y_labels[:, i]
        #     pt_list = y_preds[:, i]

        #     tp, fn, fp, tn = confusion(
        #         device, gt_list, pt_list, thresholds, approx)

        #     precision = tp/(tp+fp+EPS)
        #     recall = tp/(tp+fn+EPS)
        #     temp_f1 = torch.mean(2 * (precision * recall) /
        #                          (precision + recall + EPS))

        #     # tensor(0.5000, device='cuda:0', grad_fn=<MeanBackward0>)
        #     mean_f1s[i] = temp_f1


        # print("Correct Mean F1: {}".format(mean_f1s.mean()))
        # loss = 1 - mean_f1s.mean()
        loss = 1 - temp_f1.mean()
        return loss
    return loss






def xmean_f1_approx_loss_on(device, y_labels=None, y_preds=None, thresholds=torch.arange(0.1, 1, 0.1), approx=linear_approx()):
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

        y_labels = y_labels.to(device)
        y_preds = y_preds.to(device)
        thresholds = torch.Tensor([0.5, 0.6]).to(device)
        # thresholds = torch.arange(0.1, 1, 0.1).to(device)
        for i in range(classes):
            print("RUNNING FOR CLASS {}".format(i))
            gt_list = y_labels[:, i]
            pt_list = y_preds[:, i]
    
            tp, fn, fp, tn = confusion(device, gt_list, pt_list, thresholds, approx)
            # if i == 0: 
                # print("OLD TP")
                # tp, fn, fp, tn = confusion(device, gt_list, pt_list, thresholds, approx)
                # print(tp)
            precision = tp/(tp+fp+EPS)
            recall = tp/(tp+fn+EPS)
            temp_f1 = torch.mean(2 * (precision * recall) /
                                 (precision + recall + EPS))

            # tensor(0.5000, device='cuda:0', grad_fn=<MeanBackward0>)
            mean_f1s[i] = temp_f1

        # trying other method 
        tp, fn, fp, tn = confusion_adj(
            device, y_labels, y_preds, thresholds, approx)
        print("NEW TP:----")
        print(tp[:, 0])
        precision = tp/(tp+fp+EPS)
        recall = tp/(tp+fn+EPS)
        temp_f1 = torch.mean(2 * (precision * recall) /
                             (precision + recall + EPS), axis=0)
        # print("New Mean f1: {}".format(temp_f1))
        # print("New Loss:{}".format(1-temp_f1.mean()))
        loss = 1 - mean_f1s.mean()
        # print("OG Mean F1: {}".format(mean_f1s))
        # print("OG LOSS: {}".format(loss))
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
        mean_f1s = torch.zeros(classes, dtype=torch.float32).to(device)
        x = 0
        for i in range(classes):
            gt_list = y_labels[:, i].to(device)
            pt_list = y_preds[:, i].to(device)
            thresholds = threshold.to(device)

            tp, fn, fp, tn = confusion(
                device, gt_list, pt_list, thresholds, approx)

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



# def mean_f1_approx_loss_on(device, y_labels=None, y_preds=None, thresholds=torch.arange(0.1, 1, 0.1), approx=linear_approx()):
#     thresholds = thresholds.to(device)

#     def loss(y_labels, y_preds):
#         """
#         Approximate F1:
#             - Linear interpolated Heaviside function
#             - Harmonic mean of precision and recall
#             - Mean over a range of thresholds
#         Args:
#             y_labels: one-hot encoded label, i.e. 2 -> [0, 0, 1]
#             y_preds: softmaxed predictions
#         """
#         start = time.time()
#         classes = len(y_labels[0])
#         mean_f1s = torch.zeros(classes, dtype=torch.float32).to(device)
#         print("labels shape:{}".format(y_labels.t().shape))
#         print("preds shape:{}".format(y_preds.t().shape))

        
#         y_labels_t = y_labels.t()
#         y_preds_t = y_preds.t()

#         sample = outer_wrapper(device, y_labels_t, y_preds_t)
#         print("SAMPLE: {}".format(sample))

#         print_once = True
#         for i in range(classes):
#             gt_list = y_labels[:, i].to(device)
#             pt_list = y_preds[:, i].to(device)

#             thresholds = torch.arange(0.1, 1, 0.1).to(device)
#             tp, fn, fp, tn = confusion(
#                 device, gt_list, pt_list, thresholds, approx)
#             if print_once: 
#                 print("correct tp:{}".format(tp.shape))
#                 print_once = False
#             precision = tp/(tp+fp+EPS)
#             recall = tp/(tp+fn+EPS)
#             temp_f1 = torch.mean(2 * (precision * recall) /
#                                  (precision + recall + EPS))
#             mean_f1s[i] = temp_f1
#         # sample = torch.vmap(calc_heaviside_mean_f1, device, y_labels, y_preds)
#         # print(sample)
#         print("correct mean_f1: {}".format(mean_f1s.shape))
#         # print(sample)
#         # print("after one loop: {}".format(time.time() - start)
#         # print("OTHER CALC: {}".format(mean_f1s.mean()))
#         loss = 1 - mean_f1s.mean()
#         # loss = 1 - sample
#         return loss
#     return loss




# def mean_f1_approx_loss_on(device, y_labels=None, y_preds=None, thresholds=torch.arange(0.1, 1, 0.1), approx=linear_approx()):
#     thresholds = thresholds.to(device)

#     def loss(y_labels, y_preds):
#         """
#         Approximate F1:
#             - Linear interpolated Heaviside function
#             - Harmonic mean of precision and recall
#             - Mean over a range of thresholds
#         Args:
#             y_labels: one-hot encoded label, i.e. 2 -> [0, 0, 1]
#             y_preds: softmaxed predictions
#         """
#         start = time.time() 
#         classes = len(y_labels[0])
#         mean_f1s = torch.zeros(classes, dtype=torch.float32).to(device)

#         print("shape:{}".format(y_labels.shape))

#         for i in range(classes):
#             gt_list = y_labels[:, i].to(device)
#             pt_list = y_preds[:, i].to(device)

#             thresholds = torch.arange(0.1, 1, 0.1).to(device)
#             mid = time.time() 
#             tp, fn, fp, tn = confusion(
#                 device, gt_list, pt_list, thresholds, approx)
#             # print("WITHIN LOOP: after getting confusion metrics:{}".format(mid-time.time()))
#             precision = tp/(tp+fp+EPS)
#             recall = tp/(tp+fn+EPS)
#             temp_f1 = torch.mean(2 * (precision * recall) /
#                                  (precision + recall + EPS))
#             mean_f1s[i] = temp_f1

#         print("after one loop: {}".format(time.time() - start))
#         loss = 1 - mean_f1s.mean()
#         return loss
#     return loss



'''

RUNNING FOR CLASS 0
OG XS:tensor([[0.0929, 0.0929, 0.0929,  ..., 0.0929, 0.0929, 0.0929],
        [0.0961, 0.0961, 0.0961,  ..., 0.0961, 0.0961, 0.0961],
        [0.0941, 0.0941, 0.0941,  ..., 0.0941, 0.0941, 0.0941],
        ...,
        [0.0940, 0.0940, 0.0940,  ..., 0.0940, 0.0940, 0.0940],
        [0.0949, 0.0949, 0.0949,  ..., 0.0949, 0.0949, 0.0949],
        [0.0918, 0.0918, 0.0918,  ..., 0.0918, 0.0918, 0.0918]],
       device='cuda:1')


VALUE OF 0th' class new XS:tensor([[0.0929, 0.0981, 0.0977,  ..., 0.1068, 0.1004, 0.9061],
        [0.0961, 0.0954, 0.0974,  ..., 0.1072, 0.0974, 0.0940],
        [0.0941, 0.0961, 0.0968,  ..., 0.1024, 0.0999, 0.1021],
        ...,
        [0.0940, 0.0950, 0.0988,  ..., 0.1063, 0.0981, 0.9021],
        [0.0949, 0.0931, 0.0978,  ..., 0.1072, 0.1003, 0.0937],
        [0.0918, 0.0972, 0.9013,  ..., 0.1056, 0.1003, 0.0936]],
       device='cuda:1')
'''
