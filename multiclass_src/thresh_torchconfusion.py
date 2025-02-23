import torch

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
        conditions = torch.where(cm1, torch.tensor([1.0]), torch.where(
            cm3, torch.tensor([3.0]), torch.tensor([2.0])))
        print('x', x)
        print('t', t)
        print('conditions', conditions)
        # print(f"m1 = {d}/{t}-{tt}/2")
        # print(f"m2 = (1-2*{d})/({tt}+EPS)")
        # print(f"m3 = {d}/(1-{t}-{tt}/2)")
    res = torch.where(cm1, m1*x, torch.where(cm3, m3*x +
                                             (1-d-m3*(t+tt/2)), m2*(x-t)+0.5))
    if debug:
        print('res', res)
    return res


def heaviside_agg(xs, thresholds, agg):
    ''' xs.shape: [batchsize, thresholds]
        thresholds.shape: [batchsize, thresholds]
        approx: linear_approx or approximation function to use
        '''
    # print("threshold shape: {}".format(thresholds.shape))
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

    # just fills in the areas where we have either a 0 or 1 value. 
    thresh = torch.where(thresh == 0.0, torch.tensor([0.01], device=thresh.device),
                         torch.where(thresh == 1.0, torch.tensor([0.99], device=thresh.device), thresh))

    gt_t = torch.reshape(torch.repeat_interleave(gt, thresh.shape[0]), (-1, thresh.shape[0]))
    pt_t = torch.reshape(torch.repeat_interleave(pt, thresh.shape[0]), (-1, thresh.shape[0]))
    
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
    thresh = torch.where(thresh == 0.0, torch.tensor([0.01], device=thresh.device),
                         torch.where(thresh == 1.0, torch.tensor([0.99], device=thresh.device), thresh))

    gt_t = torch.reshape(torch.repeat_interleave(gt, thresh.shape[0]), (-1, thresh.shape[0]))
    pt_t = torch.reshape(torch.repeat_interleave(pt, thresh.shape[0]), (-1, thresh.shape[0]))

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
    gt_t = torch.reshape(torch.repeat_interleave(gt, thresh.shape[0]), (-1, thresh.shape[0]))
    pt_t = torch.reshape(torch.repeat_interleave(pt, thresh.shape[0]), (-1, thresh.shape[0]))

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
    gt_t = torch.reshape(torch.repeat_interleave(gt, thresh.shape[0]), (-1, thresh.shape[0]))
    pt_t = torch.reshape(torch.repeat_interleave(pt, thresh.shape[0]), (-1, thresh.shape[0]))

    condition = (gt_t == 1) & (pt_t < thresh)
    xs = torch.where(condition, pt_t, 1-pt_t)
    thresholds = torch.where(condition, thresh, 1-thresh)
    return heaviside_agg(xs, thresholds, agg)


def confusion(gt, pt, thresholds, agg='sum'):
    # 'tp', 'fn', 'fp', 'tn'
    # print("threshold: {}".format(thresholds))
    tp = l_tp(gt, pt, thresholds, agg)
    fn = l_fn(gt, pt, thresholds, agg)
    fp = l_fp(gt, pt, thresholds, agg)
    tn = l_tn(gt, pt, thresholds, agg)
    return tp, fn, fp, tn


def mean_f1_approx_loss_on(device, threshold, y_labels=None, y_preds=None):
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
        class_f1 = [0]*classes
        class_losses = [0]*classes
        mean_f1s = torch.zeros(classes, dtype=torch.float32).to(device)

        for i in range(classes):
            gt_list = torch.Tensor([x[i] for x in y_labels]).to(device)
            pt_list = y_preds[:, i].to(device)
            thresholds = threshold.to(device)
            
            tp, fn, fp, tn = confusion(gt_list, pt_list, thresholds)
            
            precision = tp/(tp+fp+EPS)
            recall = tp/(tp+fn+EPS)
            temp_f1 = torch.mean(2 * (precision * recall) /
                                 (precision + recall + EPS))
            mean_f1s[i] = temp_f1

            # geting the average across all thresholds, holds array of just the raw values.
            class_f1[i] = temp_f1.detach().item()
            # for each class, store the loss 
            class_losses[i] = 1 - temp_f1
        loss = 1 - mean_f1s.mean()
        return loss, class_losses 
    return loss


def st_mean_f1_approx_loss_on(device, y_labels=None, y_preds=None):
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
        for i in range(classes):
            gt_list = torch.Tensor([x[i] for x in y_labels]).to(device)
            pt_list = y_preds[:, i].to(device)
            thresholds = threshold.to(device)

            tp, fn, fp, tn = confusion(gt_list, pt_list, thresholds)

            # print("TP:{}".format(tp))
            precision = tp/(tp+fp+EPS)
            recall = tp/(tp+fn+EPS)
            temp_f1 = torch.mean(2 * (precision * recall) /
                                 (precision + recall + EPS))
            mean_f1s[i] = temp_f1


        loss = 1 - mean_f1s.mean()
        return loss
    return loss
