import torch

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
        print(f"  m1 = {d}/{t}-{tt}/2")
        print(f"  m2 = (1-2*{d})/({tt}+EPS)")
        print(f"  m3 = {d}/(1-{t}-{tt}/2)")

    res = torch.where(cm1, m1*x,
        torch.where(cm3, m3*x + (1-d-m3*(t+tt/2)),
            m2*(x-t)+0.5))
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
    thresh = torch.where(thresh==0.0, torch.tensor([0.01]),
            torch.where(thresh==1.0, torch.tensor([0.99]), thresh))

    gt_t = torch.reshape(
        torch.repeat_interleave(gt, thresh.shape[0]), 
        (-1, thresh.shape[0])
    )
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

    # filling in the places where it's 0 or 1. 
    thresh = torch.where(
        thresh==0.0, 
        torch.tensor([0.01], device=thresh.device),
        torch.where(
            thresh==1.0, 
            torch.tensor([0.99], device=thresh.device), thresh)
    )

    gt_t = torch.reshape(torch.repeat_interleave(
                            gt,
                            thresh.shape[0]
                        ), 
            (-1, thresh.shape[0])
            )
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
    thresh = torch.where(thresh==0.0, torch.tensor([0.01], device=thresh.device),
            torch.where(thresh==1.0, torch.tensor([0.99], device=thresh.device), thresh))
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
    thresh = torch.where(thresh==0.0, torch.tensor([0.01], device=thresh.device),
            torch.where(thresh==1.0, torch.tensor([0.99], device=thresh.device), thresh))
    gt_t = torch.reshape(torch.repeat_interleave(gt, thresh.shape[0]), (-1, thresh.shape[0]))
    pt_t = torch.reshape(torch.repeat_interleave(pt, thresh.shape[0]), (-1, thresh.shape[0]))
    condition = (gt_t == 1) & (pt_t < thresh)
    xs = torch.where(condition, pt_t, 1-pt_t)
    thresholds = torch.where(condition, thresh, 1-thresh)
    return heaviside_agg(xs, thresholds, agg)

# pt = predicted 
# gt = ground truth 
def confusion(gt, pt, thresholds, agg='sum'):
    print("gt: {}".format(gt))
    print("pt: {}".format(pt))
    # 'tp', 'fn', 'fp', 'tn'
    tp = l_tp(gt, pt, thresholds, agg)
    fn = l_fn(gt, pt, thresholds, agg)
    fp = l_fp(gt, pt, thresholds, agg)
    tn = l_tn(gt, pt, thresholds, agg)
    # print("tp: {} | fn: {} | fp: {} | tn: {}".format(
    #     tp.detach().numpy(), fn.detach().numpy(), fp.detach().numpy(), tn.detach().numpy())
    #     )
    return tp, fn, fp, tn
