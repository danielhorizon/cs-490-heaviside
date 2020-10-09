import time 
import torch
import click 
import logging
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchconfusion import confusion
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

from ap_perf import PerformanceMetric, MetricLayer
from ap_perf.metric import CM_Value

from torch.autograd import Variable
from keras.utils import to_categorical



EPS = 1e-7
_IRIS_DATA_PATH = "../data/iris.csv"

class Model(nn.Module):
    def __init__(self, input_features=4, hidden_layer1=8, hidden_layer2=12, output_features=3):
        super().__init__()
        self.fc1 = nn.Linear(input_features, hidden_layer1)
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.output = nn.Linear(hidden_layer2, output_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x

# metric definition
class Fbeta(PerformanceMetric):
    def __init__(self, beta):
        self.beta = beta

    def define(self, C):
        return ((1 + self.beta ** 2) * C.tp) / ((self.beta ** 2) * C.ap + C.pp)

# metric definition
class AccuracyMetric(PerformanceMetric):
    def define(self, C):
        return (C.tp + C.tn) / C.all


def mean_f1_approx_loss_on(thresholds=torch.arange(0.1, 1, 0.1)):
    def loss(pt, gt):
        """Approximate F1:
            - Linear interpolated Heaviside function
            - Harmonic mean of precision and recall
            - Mean over a range of thresholds
        """
        classes = pt.shape[1] if len(pt.shape) == 2 else 1
        mean_f1s = torch.zeros(classes, dtype=torch.float32)
        # mean over all classes
        for i in range(classes):
            thresholds = torch.arange(0.1, 1, 0.1)
            tp, fn, fp, _ = confusion(gt, pt[:,i] if classes > 1 else pt, thresholds)
            precision = tp/(tp+fp+EPS)
            recall = tp/(tp+fn+EPS)
            mean_f1s[i] = torch.mean(2 * (precision * recall) / (precision + recall + EPS))
        loss = 1 - mean_f1s.mean()
        return loss
    return loss


def mean_accuracy_approx_loss_on(thresholds=torch.arange(0.1, 1, 0.1)):
    def loss(pt, gt):
        """Approximate Accuracy:
            - Linear interpolated Heaviside function
            - (TP + TN) / (TP + TN + FP + FN)
            - Mean over a range of thresholds
        """
        classes = pt.shape[1] if len(pt.shape) == 2 else 1
        mean_accs = torch.zeros(classes, dtype=torch.float32)
        # mean over all classes
        for i in range(classes):
            tp, fn, fp, tn = confusion(gt, pt[:,i] if classes > 1 else pt, thresholds)
            mean_accs[i] = torch.mean((tp + tn) / (tp + tn + fp + fn))
        loss = 1 - mean_accs.mean()
        return loss
    return loss


def area(x,y):
    ''' area under curve via trapezoidal rule'''
    direction = 1
    # the following is equivalent to: dx = np.diff(x)
    dx = x[1:] - x[:-1]
    if torch.any(dx < 0):
        if torch.all(dx <= 0):
            direction = -1
        else:
            logging.warn("x is neither increasing nor decreasing\nx: {}\ndx: {}.".format(x, dx))
            return 0
    return direction * torch.trapz(y, x)


def mean_auroc_approx_loss_on(linspacing=11):
    def loss(pt, gt):
        """Approximate auroc:
            - Linear interpolated Heaviside function
            - roc (11-point approximation)
            - integrate via trapezoidal rule under curve
        """
        classes = pt.shape[1] if len(pt.shape) == 2 else 1
        thresholds = torch.linspace(0, 1, linspacing)
        areas = []
        # mean over all classes
        for i in range(classes):
            tp, fn, fp, tn = confusion(gt, pt[:,i] if classes > 1 else pt, thresholds)
            fpr = fp/(fp+tn+EPS)
            tpr = tp/(tp+fn+EPS)
            a = area(fpr, tpr)
            if a > 0:
                areas.append(a)
        loss = 1 - torch.stack(areas).mean()
        return loss
    return loss


def threshold_pred(y_pred, t):
    return (y_pred > t).float()


def train(args):
    # initialize metric
    f1_score = Fbeta(1)
    f1_score.initialize()
    f1_score.enforce_special_case_positive()

    # accuracy metric
    accm = AccuracyMetric()
    accm.initialize()

    threshold = 0.5

    # create a model and criterion layer
    sigmoid_out = False
    if args.loss in ['approx-f1', 'approx-accuracy', 'approx-auroc', 'approx-ap', 'auc-roc']:
        sigmoid_out = True
    model = Net(input_dim, sigmoid_out).to(device)

    # set run timestamp or load from args
    now = int(time.time())
    if args.initial_weights:
        now = args.initial_weights

    Path(model_path(args)).mkdir(parents=True, exist_ok=True)
    run_name = f"{args.dataset}-{args.loss}-batch_{args.batch_size}-lr_{args.lr}_{now}"
    log_path = '/'.join([model_path(args), f"{run_name}.log"])
    initial_model_file_path = '/'.join([model_path(args), '{}_initial.pth'.format(now)])

    # load or save initial weights
    if args.initial_weights:
        logging.info(f"[{now}] loading {initial_model_file_path}")
        model.load_state_dict(torch.load(initial_model_file_path))
    else:
        # persist the initial weights for future use
        torch.save(model.state_dict(), initial_model_file_path)

    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'ap-perf-f1':
        threshold = 0.0
        criterion = MetricLayer(f1_score).to(device)
    elif args.loss == 'approx-f1':
        criterion = mean_f1_approx_loss_on(device, thresholds=torch.tensor([0.5]))
    elif args.loss == 'approx-accuracy':
        criterion = mean_accuracy_approx_loss_on(device, thresholds=torch.tensor([0.5]))
    elif args.loss == 'approx-auroc':
        criterion = mean_auroc_approx_loss_on(device)
    elif args.loss == 'approx-ap':
        criterion = mean_ap_approx_loss_on(device)
    elif args.loss == 'auc-roc':
        criterion = roc_auc_score(device)
    else:
        raise RuntimeError("Unknown loss {}".format(args.loss))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    patience = args.early_stopping_patience

    tensorboard_path = '/'.join([args.output, 'running', args.experiment, 'tensorboard', run_name])
    writer = SummaryWriter(tensorboard_path)

    # early stopping
    early_stopping = False
    best_f1 = None
    best_f1_apperf = 0

    best_test = {
        'now': now,
        'loss': args.loss,
        'accuracy_05_score':0,
        'f1_05_score':0,
        'ap_05_score':0,
        'auroc_05_score':0,
        'accuracy_mean_score': 0,
        'f1_mean_score': 0,
        'ap_mean_score':0,
        'auroc_mean_score':0,
    }

    for epoch in range(args.epochs):
        if early_stopping:
            logging.info("[{}] Early Stopping at Epoch {}/{}".format(now, epoch, args.epochs))
            logging.info("  [val best f1] apperf: {:.4f} | ignite: {:.4f}".format(best_f1_apperf, best_f1))
            break

        data_cm = CM_Value(np.array([]),np.array([]))

        losses = []
        accuracies = []
        f1s = []
        aps = []
        rocs = []

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            model.train()
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            losses.append(loss)
            loss.backward()
            optimizer.step()

            ## check prediction
            model.eval()    # switch to evaluation
            y_pred = model(inputs)
            y_pred_thresh = (y_pred >= threshold).float()
            np_pred = y_pred_thresh.cpu().numpy()
            np_labels = labels.cpu().numpy()
            batch_cm = CM_Value(np_pred, np_labels)
            data_cm = add_cm_val(data_cm, batch_cm)
            # sklearn.metrics to tensorboard
            accuracies.append(metrics.accuracy_score(np_labels, np_pred))
            f1s.append(metrics.f1_score(np_labels,np_pred))
            aps.append(metrics.average_precision_score(np_labels,np_pred))
            # undefined if predicting only 1 value
            # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/_ranking.py#L224
            if len(np.unique(np_labels)) == 2:
                rocs.append(metrics.roc_auc_score(np_labels,np_pred))

        acc_val = compute_metric_from_cm(accm, data_cm)
        f1_val = compute_metric_from_cm(f1_score, data_cm)

        mloss = np.array(loss.cpu().detach()).mean()
        writer.add_scalar('loss', mloss, epoch)
        writer.add_scalar('train/accuracy', np.array(acc_val).mean(), epoch)
        writer.add_scalar('train/f1', np.array(f1_val).mean(), epoch)
        writer.add_scalar('train/ap', np.array(aps).mean(), epoch)
        writer.add_scalar('train/auroc', np.array(rocs).mean(), epoch)

        logging.info("Train - Epoch ({}): Loss: {:.4f} Accuracy: {:.4f} | F1: {:.4f}".format(epoch, mloss, acc_val, f1_val))


        ### Validation
        model.eval()
        with torch.no_grad():
            data_cm = CM_Value(np.array([]),np.array([]))
            for i, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                output = model(inputs)

                ## prediction
                pred = (output >= 0).float()
                batch_cm = CM_Value(pred.cpu().numpy(), labels.cpu().numpy())
                data_cm = add_cm_val(data_cm, batch_cm)

            acc_val = compute_metric_from_cm(accm, data_cm)
            f1_val_apperf = compute_metric_from_cm(f1_score, data_cm)
            writer.add_scalar('val_ap_perf/accuracy', acc_val, epoch)
            writer.add_scalar('val_ap_perf/f1', f1_val_apperf, epoch)
            if best_f1_apperf < f1_val_apperf:
                best_f1_apperf = f1_val_apperf

            logging.info("Val - Epoch ({}): Accuracy: {:.4f} | F1: {:.4f}".format(epoch, acc_val, f1_val))

            # double check val metrics
            inferencer = inference_engine(model, device, threshold=threshold)
            result_state = inferencer.run(val_loader)
            logging.info("  val: {}".format(result_state.metrics))
            writer.add_scalar('val/accuracy', result_state.metrics['accuracy'], epoch)
            writer.add_scalar('val/f1', result_state.metrics['f1'], epoch)
            writer.add_scalar('val/ap', result_state.metrics['ap'], epoch)
            writer.add_scalar('val/auroc', result_state.metrics['auroc'], epoch)
            f1_val_ignite = result_state.metrics['f1']

            # check early stopping per epoch
            patience -= 1
            if best_f1 is None or best_f1 < f1_val_ignite:
                # save the best model
                model_file_path = '/'.join([model_path(args), '{}_best_model_{}_{}_{}={}.pth'.format(now, epoch, args.dataset, args.loss, f1_val_ignite)])
                torch.save(model, model_file_path)
                logging.info("Saving best model to {}".format(model_file_path))
                best_f1 = f1_val_ignite
                patience = args.early_stopping_patience
                # check test set results
                results = inference_over_range(model, device, test_loader)
                # values correspond to the thresholds: np.arange(0.1,1,0.1), so index 4 has t=0.5
                accuracies = [r.metrics['accuracy'] for r in results]
                test_f1s = [r.metrics['f1'] for r in results]
                aps = [r.metrics['ap'] for r in results]
                aurocs = [r.metrics['auroc'] for r in results]
                # record the best to print at the end
                if best_test['accuracy_05_score'] < accuracies[4]:
                    best_test['accuracy_05_score'] = accuracies[4]
                    best_test['accuracy_05_model_file'] = model_file_path
                if best_test['f1_05_score'] < test_f1s[4]:
                    best_test['f1_05_score'] = test_f1s[4]
                    best_test['f1_05_model_file'] = model_file_path
                if best_test['ap_05_score'] < aps[4]:
                    best_test['ap_05_score'] = aps[4]
                    best_test['ap_05_model_file'] = model_file_path
                if best_test['auroc_05_score'] < aurocs[4]:
                    best_test['auroc_05_score'] = aurocs[4]
                    best_test['auroc_05_model_file'] = model_file_path
                mean_accuracy = np.mean(accuracies)
                mean_f1 = np.mean(test_f1s)
                mean_ap = np.mean(aps)
                mean_auroc = np.mean(aurocs)
                if best_test['accuracy_mean_score'] < mean_accuracy:
                    best_test['accuracy_mean_score'] = mean_accuracy
                    best_test['accuracy_mean_model_file'] = model_file_path
                if best_test['f1_mean_score'] < mean_f1:
                    best_test['f1_mean_score'] = mean_f1
                    best_test['f1_mean_model_file'] = model_file_path
                if best_test['ap_mean_score'] < mean_ap:
                    best_test['ap_mean_score'] = mean_ap
                    best_test['ap_mean_model_file'] = model_file_path
                if best_test['auroc_mean_score'] < mean_auroc:
                    best_test['auroc_mean_score'] = mean_auroc
                    best_test['auroc_mean_model_file'] = model_file_path
                # write to tensorboard
                writer.add_scalar('test/accuracy_05', accuracies[4], epoch)
                writer.add_scalar('test/f1_05', test_f1s[4], epoch)
                writer.add_scalar('test/ap_05', aurocs[4], epoch)
                writer.add_scalar('test/auroc_05', aurocs[4], epoch)
                writer.add_scalar('test/accuracy_mean', mean_accuracy, epoch)
                writer.add_scalar('test/f1_mean', mean_f1, epoch)
                writer.add_scalar('test/ap_mean', mean_auroc, epoch)
                writer.add_scalar('test/auroc_mean', mean_auroc, epoch)
            logging.info(f"[{now}] {args.loss}, patience: {patience}")
            if patience <= 0:
                early_stopping = True

    logging.info(f"{args.experiment} {now}")
    logging.info(best_test)
    pd.DataFrame({k: [v] for k, v in best_test.items()}).to_csv('/'.join([model_path(args), f"{run_name}.csv"]))
    return now


def test(args):
    now = int(time.time())
    if args.initial_weights:
        now = args.initial_weights

    if torch.cuda.is_available():
      device = f"cuda:{args.gpu}"
    else:
      device = "cpu"

    threshold = 0.5
    if args.loss == 'ap-perf-f1':
        threshold = 0.0
    ds = dataset_from_name(args.dataset)()
    dataparams = {'batch_size': args.batch_size,
                  'shuffle': True,
                  'num_workers': 1}

    trainset = Dataset(ds['train'])
    train_loader = DataLoader(trainset, **dataparams)
    validationset = Dataset(ds['val'])
    val_loader = DataLoader(validationset, **dataparams)
    testset = Dataset(ds['test'])
    test_loader = DataLoader(testset, **dataparams)

    input_dim = ds['train']['X'][0].shape[0]

    best_model_path = None
    maxval = None
    globpath = '/'.join([args.output, 'running', args.experiment, dataset_name, args.loss, "{}_best_model_*_{}*.pth".format(now, args.loss)])
    for path in glob.iglob(globpath):
        val = float(path.split('=')[-1].split('.pth')[0])
        print(val)
        if maxval is None or val > maxval:
            maxval = val
            best_model_path = path
    print("loading: {}".format(best_model_path))
    #best_model.load_state_dict(torch.load(best_model_path))
    best_model = torch.load(best_model_path)

    inferencer = inference_engine(model, device, threshold=threshold)

    #ProgressBar(desc="Inference").attach(inferencer)

    result_state = inferencer.run(test_loader)
    print(result_state.metrics)

    return now


def load_iris(): 
    iris = pd.read_csv(_IRIS_DATA_PATH)
    mappings = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
    }
    iris["species"] = iris["species"].apply(lambda x: mappings[x])

    X = iris.drop("species", axis=1).values
    y = iris["species"].values
    input_size = X.shape[-1]
    cats = np.sum(np.unique(y)).astype(int)

    print("no. of samples: {}".format(X.shape[0]))       # 150 
    print("no. of attributes: {}".format(input_size))    # 4 
    print("no. of categories: {}".format(cats))          # 3 

    # purely for plotting
    # df = iris.iloc[:, 0:4]
    # fig, ax = plt.subplots(figsize=(12, 12), dpi=150)
    # pd.plotting.scatter_matrix(df, figsize=(12, 12), c=y, s=200, alpha=1, ax=ax)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    # train 
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)

    # test 
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    return {
        'train': {
            'X': X_train,
            'y': y_train
        },
        'test': {
            'X': X_test,
            'y': y_test
        },
    }


def train_iris(data_splits, loss_metric): 
    X_train, y_train = data_splits['train']['X'], data_splits['train']['y']
    X_test, y_test = data_splits['test']['X'], data_splits['test']['y']

    # initialize metric
    f1_score = Fbeta(1)
    f1_score.initialize()
    f1_score.enforce_special_case_positive()

    # accuracy metric
    accm = AccuracyMetric()
    accm.initialize()

    # initialize 
    epochs = 100
    losses = []
    model = Model()

    # criterion 
    if loss_metric == "ce":
        criterion = nn.CrossEntropyLoss()
    elif loss_metric == "approx-f1": 
        criterion = mean_f1_approx_loss_on(thresholds=torch.tensor([0.5]))
    elif loss_metric == 'approx-accuracy':
        criterion = mean_accuracy_approx_loss_on(thresholds=torch.tensor([0.5]))
    elif loss_metric == 'approx-auroc':
        criterion = mean_auroc_approx_loss_on()
    else:
        raise RuntimeError("Unknown loss {}".format(loss_metric))
    
    # setting optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for i in range(epochs):
        y_pred = model.forward(X_train)
        loss = criterion(y_pred, y_train)
        losses.append(loss)
        print(f'epoch: {i:2}  loss: {loss.item():10.8f}')

        # backprop, updating weights and biases
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

@click.command() 
@click.option("--loss", required=True)
def run(loss): 
    data_splits = load_iris()
    train_iris(data_splits, loss_metric=loss)

def main():
    run() 
    
if __name__ == '__main__':
    main()

