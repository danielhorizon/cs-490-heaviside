import json 
import os 
import torch

from sklearn.metrics import confusion_matrix

EPS = 1e-7


def record_results(best_test):
    # reading in the data from the existing file.
    with open("results.json", "r+") as f:
        data = json.load(f)
        data.append(best_test)
        f.close()
        # f.seek(0)
    
    with open("results.json", "w") as outfile: 
        json.dump(data, outfile)
    return 


def get_confusion(gt, pt, class_value=None, threshold=None):
    """ Getting tp, fp, fn, tn for a class. 
    """
    # based on threshold 
    pt = [x for x in pt if x >= threshold]
    tp = len([a for a, p in zip(gt, pt) if a == p and p == class_value])
    fp = len([a for a, p in zip(gt, pt) if a != p and p == class_value])
    fn = len([a for a, p in zip(gt, pt) if a != p and a == class_value])
    tn = len([a for a, p in zip(gt, pt) if a == p and p != class_value])
    return tp, fp, fn, tn


def f1_evaluation(y_labels=None, y_preds=None, threshold=0.1):
    classes = len(y_labels[0])
    mean_f1s = torch.zeros(classes, dtype=torch.float32)

    '''
    y_labels = tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]])
    y_preds = tensor([[0.0981, 0.0968, 0.0977, 0.0869, 0.1180, 0.1081, 0.0972, 0.0919, 0.1003, 0.1050]])
    '''

    for i in range(classes):
        # GT_T: tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        # PT_T: tensor([[0.9921, 0.9921, 0.9921, 0.9921, 0.9921, 0.9921, 0.9921, 0.9921, 0.9921]],
        gt_list = torch.Tensor([x[i] for x in y_labels])
        pt_list = y_preds[:, i]
        pt_list = torch.Tensor([1 if x >= threshold else 0 for x in pt_list])
        tn, fp, fn, tp = confusion_matrix(y_true=gt_list, y_pred=pt_list, labels=[0,1]).ravel() 

        # converting to tensors 
        tp, fn, fp, tn = torch.tensor([tp]), torch.tensor([fn]), torch.tensor([fp]), torch.tensor([tn])
        precision = tp/(tp+fp+EPS)
        recall = tp/(tp+fn+EPS)
        temp_f1 = torch.mean(2 * (precision * recall) / (precision + recall + EPS))
        print("temp f1:{}".format(temp_f1))
        mean_f1s[i] = temp_f1

    return mean_f1s.mean()


def actual_evaluation_f1(device, y_labels=None, y_preds=None, threshold=None):
    classes = len(y_labels[0])
    mean_f1s = torch.zeros(classes, dtype=torch.float32)

    '''
    y_labels = tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]])
    y_preds = tensor([[0.0981, 0.0968, 0.0977, 0.0869, 0.1180, 0.1081, 0.0972, 0.0919, 0.1003, 0.1050]])
    '''

    print("LABELS:{}".format(y_labels))
    print("PREDS: {}".format(y_preds))

    for i in range(classes):
        gt_list = torch.Tensor([x[i] for x in y_labels]).to(device)
        pt_list = y_preds[:, i]

        # GT LIST:tensor([0., 0., 1.,  ..., 0., 1., 0.], device='cuda:0')
        # PT LIST: tensor([0.1047, 0.1021, 0.1016,  ..., 0.1004, 0.1035, 0.1009], device='cuda:0', grad_fn= < SelectBackward > )

        # print("GT LIST:{}".format(gt_list))
        # print("PT LIST:{}".format(pt_list))
        # tensor([1., 1., 1.,  ..., 1., 1., 1.])
        pt_list = torch.Tensor([1 if x >= threshold else 0 for x in pt_list])

        tn, fp, fn, tp = confusion_matrix(y_true=gt_list.cpu().numpy(
        ), y_pred=pt_list.cpu().numpy(), labels=[0, 1]).ravel()

        # converting to tensors
        tp, fn, fp, tn = torch.tensor([tp]).to(device), torch.tensor([fn]).to(
            device), torch.tensor([fp]).to(device), torch.tensor([tn]).to(device)
        precision = tp/(tp+fp+EPS)
        recall = tp/(tp+fn+EPS)
        temp_f1 = torch.mean(2 * (precision * recall) /
                             (precision + recall + EPS))
        mean_f1s[i] = temp_f1

    # return class wise f1, and the mean of the f1s.
    return mean_f1s, mean_f1s.mean()


if __name__ == "__main__":
    best_test = {
        "best-epoch": 0,
        "imbalanced": True,
        "learning_rate": 0.001,
        "loss": False,
        "loss_metric": "approx-f1",
        "run_name": "testoutfile",
        "seed": 45,
        "test_accuracy": 0,
        "test_wt_f1_score": 0,
        "val_accuracy": 0,
        "val_wt_f1_score": 0
    }
    # record_results(best_test)
    y_preds = torch.tensor([[0.1011, 0.1012, 0.0960, 0.0892, 0.1059, 0.1078, 0.0975, 0.0932, 0.1025, 0.1055], 
                            [0.1002, 0.0996, 0.0904, 0.0821, 0.1060, 0.1512, 0.0844, 0.0838, 0.1184, 0.0840], 
                            [0.1002, 0.0996, 0.0904, 0.0821, 0.1060, 0.1512, 0.0844, 0.0838, 0.1184, 0.0840]])
    y_labels = torch.tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.], 
                    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]])
    print(f1_evaluation(y_labels=y_labels, y_preds=y_preds))


