import os
import io
import json
import time
import click
import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

# torch imports
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# sklearn imports
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# custom imports 
# write an import to get the load_iris method from the datasets.py file in the toy folder
from datasets import *
from models import *
from constants import *
from mc_torchconfusion import *


# constants
EPS = 1e-7

def record_results(results_path, best_test, output_file):
    # reading in the data from the existing file.
    file_path = "/".join([results_path, output_file])
    if os.path.isfile(file_path):
        with open(file_path, "r+") as f:
            data = json.load(f)
            data.append(best_test)
            f.close()
        with open(file_path, "w") as outfile:
            json.dump(data, outfile)
    # if the file doesn't eixst:
    else:
        best_test = [best_test]
        with open(file_path, "w") as outfile:
            json.dump(best_test, outfile)


def evaluation_f1(device, y_labels=None, y_preds=None, threshold=None):
    classes = len(y_labels[0])
    mean_f1s = torch.zeros(classes, dtype=torch.float32)
    precisions = torch.zeros(classes, dtype=torch.float32)
    recalls = torch.zeros(classes, dtype=torch.float32)

    '''
    y_labels = tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]])
    y_preds = tensor([[0.0981, 0.0968, 0.0977, 0.0869, 0.1180, 0.1081, 0.0972, 0.0919, 0.1003, 0.1050]])
    '''
    for i in range(classes):
        gt_list = torch.Tensor([x[i] for x in y_labels]).to(device)
        pt_list = y_preds[:, i]

        pt_list = torch.Tensor([1 if x >= threshold else 0 for x in pt_list])

        tn, fp, fn, tp = confusion_matrix(y_true=gt_list.cpu().numpy(),
                                          y_pred=pt_list.cpu().numpy(), labels=[0, 1]).ravel()

        # converting to tensors
        tp, fn, fp, tn = torch.tensor([tp]).to(device), torch.tensor([fn]).to(
            device), torch.tensor([fp]).to(device), torch.tensor([tn]).to(device)
        precision = tp/(tp+fp+EPS)
        recall = tp/(tp+fn+EPS)
        temp_f1 = torch.mean(2 * (precision * recall) /
                             (precision + recall + EPS))
        mean_f1s[i] = temp_f1
        precisions[i] = precision
        recalls[i] = recall

    # return class wise f1, and the mean of the f1s.
    return mean_f1s, mean_f1s.mean(), precisions, recalls


def train_for_dataset(dataset, loss_metric, epochs, seed, experiment, cuda, batch_size, 
        patience, output_path, init_weights):
    # for gpu usage to cuda.
    if torch.cuda.is_available():
        logging.info("device = cuda :{}".format(type(cuda)))
        cuda_map = {
            "0": "cuda:0",
            "1": "cuda:1",
            "2": "cuda:2",
            "3": "cuda:3"
        }
        device = cuda_map[cuda]
    else:
        logging.info("device = cpu")
        device = "cpu"

    # setting seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # using DataSet and DataLoader
    N_CLASSES = None 
    if dataset == 'iris':
        data_splits = load_iris(seed=seed)
        dataparams = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}
        trainset = Dataset(data_splits['train'])
        validationset = Dataset(data_splits['val'])
        testset = Dataset(data_splits['test'])
        train_loader = DataLoader(trainset, **dataparams)
        val_loader = DataLoader(validationset, **dataparams)
        test_loader = DataLoader(testset, **dataparams)
        N_CLASSES = 3
        model = IrisModel().to(device)
        folder_name = 'iris'
    else:
        pass # TODO: add other datasets

    # storing metrics + experiments.
    train_count, test_count, valid_count = [0]*N_CLASSES, [0]*N_CLASSES, [0]*N_CLASSES

    # check if the folder exists 
    if not os.path.exists("{}/tensorboard/{}".format(output_path, folder_name)):
        os.makedirs("tensorboard/{}".format(folder_name))
        TENSORBOARD_PATH = "/".join([output_path, "tensorboard", folder_name, experiment])
        writer = SummaryWriter(TENSORBOARD_PATH)
        logging.info("logging tensorboard metrics to {}".format(TENSORBOARD_PATH))

    # load or save initial weights 
    if init_weights:
        logging.info("loading initial weights from {}".format(init_weights))
        model.load_state_dict(torch.load(init_weights))

    # criterion
    criterion_has_cm = False
    thresholds=torch.arange(0.1, 1, 0.1)
    if loss_metric == "ce":
        criterion = nn.CrossEntropyLoss()
    # evaluating this across thresholds, not for a single threshold. 
    # determine if we should be doing this or not...
    elif loss_metric == "approx-f1":
        criterion_has_cm = True
        criterion = mean_f1_approx_loss_on(device=device, thresholds=thresholds)
    elif loss_metric == "approx-acc":
        criterion_has_cm = True
        criterion = mean_accuracy_approx_loss_on(device=device, thresholds=thresholds)
    else:
        raise RuntimeError("unknown loss {}".format(loss_metric))

    # starting training 
    early_stopping = False
    lowest_f1_loss = None
    reset_patience = patience
        # initialization
    learning_rate = 0.001
    # for early stopping
    model_file_path = None
    best_model_in_mem = None
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    BEST_TEST = {
        "experiment": experiment,
        "best-epoch": 0,
        "loss": float('inf'),
        "test_accuracy": 0,
        "val_accuracy": 0,
        "learning_rate": learning_rate,
        "imbalanced": False, # edit this later on
        "loss_metric": loss_metric,
        "train_count": None,
        "test_count": None,
        "valid_count": None,
        "seed": seed,
        "batch_size": batch_size,
        "evaluation": None,
        "patience": patience
    }

    
    for epoch in range(epochs):
        losses = []
        if early_stopping:
            logging.info("early stopping at Epoch {}/{}".format(epoch, epochs))
            break

        accs, microf1s, macrof1s, wt_f1s = [], [], [], []
        micro_prs, macro_prs, wt_prs = [], [], []
        micro_recalls, macro_recalls, wt_recalls = [], [], []
        class_f1_scores, class_precision, class_recall = {}, {}, {}

        # starting training.
        for batch, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.type(torch.int64)
            labels = labels.to(device)
            model.train()

            # for class distribution tracking
            labels_list = labels.cpu().numpy()
            for label in labels_list:
                train_count[int(label)] += 1

            # zero grad
            optimizer.zero_grad()
            output = model(inputs)

            # if heaviside approx: 
            if criterion_has_cm:
                # convert labels to one-hot
                train_labels = torch.zeros(len(labels), len(output[0])).to(device).scatter_(
                    1, labels.unsqueeze(1), 1.).to(device)
                output = output.to(device)
                loss, tp, fn, fp, tn = criterion(y_labels=train_labels, y_preds=output)
            else:
                loss = criterion(output, labels)

            losses.append(loss)
            if loss.requires_grad:
                loss.backward()
                optimizer.step()

            # checking prediction via evaluation (for every batch)
            model.eval()
            y_pred = model(inputs)
            _, train_preds = torch.max(y_pred, 1)

            # storing metrics for each batch
            # accs = array of each batch's accuracy -> averaged at each epoch
            # this is for each batch, not for each epoch! 
            acc = accuracy_score(y_true=labels.cpu(), y_pred=train_preds.cpu())
            microf1 = f1_score(y_true=labels.cpu(), y_pred=train_preds.cpu(), average="micro")
            macrof1 = f1_score(y_true=labels.cpu(), y_pred=train_preds.cpu(), average="macro")
            w_f1 = f1_score(y_true=labels.cpu(), y_pred=train_preds.cpu(), average="weighted")
            micro_pr = precision_score(y_true=labels.cpu(), y_pred=train_preds.cpu(), average="micro")
            macro_pr = precision_score(y_true=labels.cpu(), y_pred=train_preds.cpu(), average="macro")
            weighted_pr = precision_score(y_true=labels.cpu(), y_pred=train_preds.cpu(), average="weighted")
            micro_recall = recall_score(y_true=labels.cpu(), y_pred=train_preds.cpu(), average="micro")
            macro_recall = recall_score(y_true=labels.cpu(), y_pred=train_preds.cpu(), average="macro")
            weighted_recall = recall_score(y_true=labels.cpu(), y_pred=train_preds.cpu(), average="weighted")
            # this is an array of scores
            class_f1s = f1_score(y_true=labels.cpu(), y_pred=train_preds.cpu(), average=None)
            class_re = recall_score(y_true=labels.cpu(), y_pred=train_preds.cpu(), average=None)
            class_pr = precision_score(y_true=labels.cpu(), y_pred=train_preds.cpu(), average=None)

            # storing metrics for each batch
            accs.append(acc)
            microf1s.append(microf1)
            macrof1s.append(macrof1)
            wt_f1s.append(w_f1)
            micro_prs.append(micro_pr)
            macro_prs.append(macro_pr)
            wt_prs.append(weighted_pr)
            micro_recalls.append(micro_recall)
            macro_recalls.append(macro_recall)
            wt_recalls.append(weighted_recall)
            for i in range(len(class_f1s)):
                class_f1_scores[i].append(class_f1s[i])
                class_precision[i].append(class_pr[i])
                class_recall[i].append(class_re[i])
            

        # epoch logging + loss calculation
        # https://github.com/rizalzaf/ap_perf/blob/master/examples/tabular.py
        m_loss = torch.mean(torch.stack(losses)) if torch.cuda.is_available() else np.array([x.item() for x in losses]).mean()
        m_accs = np.array(accs).mean()
        m_weightedf1s = np.array(microf1s).mean()
        m_microf1s = np.array(microf1s).mean()
        m_macrof1s = np.array(macrof1s).mean()
        logging.info("train - epoch ({}): | acc: {:.3f} | wt f1: {:.3f} | micro f1: {:.3f}| macro f1: {:.3f}".format(
            epoch, m_accs, m_weightedf1s, m_microf1s, m_macrof1s)
        )
        if criterion_has_cm:
            writer.add_scalar("train/loss", m_loss, epoch)
            writer.add_scalar("train/accuracy", m_accs, epoch)
            writer.add_scalar("train/w-f1", m_weightedf1s, epoch)
            writer.add_scalar("train/micro-f1", m_microf1s, epoch)
            writer.add_scalar("train/macro-f1", m_macrof1s, epoch)
            writer.add_scalar("train/w-recall", np.array(wt_recalls).mean(), epoch)
            writer.add_scalar("train/micro-recall", np.array(micro_recalls).mean(), epoch)
            writer.add_scalar("train/macro-recall", np.array(macro_recalls).mean(), epoch)
            writer.add_scalar("train/w-precision", np.array(wt_prs).mean(), epoch)
            writer.add_scalar("train/micro-precision", np.array(micro_prs).mean(), epoch)
            writer.add_scalar("train/macro-precision", np.array(macro_prs).mean(), epoch)

            # adding per-class f1, precision, and recall
            for i in range(N_CLASSES):
                title = "train/class-" + str(i) + "-f1"
                writer.add_scalar(title, np.array(class_f1_scores[i]).mean(), epoch)
                title = "train/class-" + str(i) + "-precision"
                writer.add_scalar(title, np.array(class_precision[i]).mean(), epoch)
                title = "train/class-" + str(i) + "-recall"
                writer.add_scalar(title, np.array(class_recall[i]).mean(), epoch)
        # basic logging regardless of loss: 
        writer.add_scalar("train/loss", m_loss, epoch)
        writer.add_scalar("train/accuracy", m_accs, epoch)
        writer.add_scalar("train/w-f1", m_weightedf1s, epoch)
        writer.add_scalar("train/micro-f1", m_microf1s, epoch)
        writer.add_scalar("train/macro-f1", m_macrof1s, epoch)

        # ---------- validation ----------
        # Calculate metrics after going through all the batches
        model.eval()
        with torch.no_grad():
            val_preds, val_labels = np.array([]), np.array([])
            valid_losses = []
            # loop through validation batches
            for batch, (inputs, labels) in enumerate(val_loader):
                labels_list = labels.cpu().numpy()
                # add to valid count
                for label in labels_list:
                    valid_count[int(label)] += 1

                inputs = inputs.to(device)
                labels = labels.to(device)

                output = model(inputs)
                _, predicted = torch.max(output, 1)

                # calculate metrics
                pred_arr = predicted.cpu().numpy()
                label_arr = labels.cpu().numpy()

                val_labels = np.concatenate([val_labels, label_arr])
                val_preds = np.concatenate([val_preds, pred_arr])

                # calculate validation loss
                labels = labels.type(torch.int64)
                if criterion_has_cm:
                    valid_labels = torch.zeros(len(labels), len(output[0])).to(device).scatter_(1, labels.unsqueeze(1), 1.).to(device)
                    output = output.to(device)
                    batch_val_loss = criterion(
                        y_labels=valid_labels, y_preds=output)
                else:
                    batch_val_loss = criterion(output, labels)
                valid_losses.append(batch_val_loss.detach().cpu().numpy())

            # after looping through all batches, calculate validation metrics
            val_acc = accuracy_score(y_true=val_labels, y_pred=val_preds)
            val_f1_micro = f1_score(y_true=val_labels, y_pred=val_preds, average='micro')
            val_f1_macro = f1_score(y_true=val_labels, y_pred=val_preds, average='macro')
            val_f1_weighted = f1_score(y_true=val_labels, y_pred=val_preds, average='weighted')
            class_val_f1 = f1_score(y_true=val_labels, y_pred=val_preds, average=None)
            class_val_pr = precision_score(y_true=val_labels, y_pred=val_preds, average=None)
            class_val_re = recall_score(y_true=val_labels, y_pred=val_preds, average=None)
            valid_loss = np.mean(valid_losses)

            if criterion_has_cm:
                writer.add_scalar("val/train-loss", valid_loss, epoch)
                writer.add_scalar("val/accuracy", val_acc, epoch)
                writer.add_scalar("val/w-f1", val_f1_weighted, epoch)
                writer.add_scalar("val/micro-f1", val_f1_micro, epoch)
                writer.add_scalar("val/macro-f1", val_f1_macro, epoch)

                # adding per-class f1, precision, and recall
                for i in range(N_CLASSES):
                    title = "val/class-" + str(i) + "-f1"
                    writer.add_scalar(title, class_val_f1[i], epoch)
                    title = "val/class-" + str(i) + "-precision"
                    writer.add_scalar(title, class_val_pr[i], epoch)
                    title = "val/class-" + str(i) + "-recall"
                    writer.add_scalar(title, class_val_re[i], epoch)

            writer.add_scalar("val/train-loss", valid_loss, epoch)
            writer.add_scalar("val/accuracy", val_acc, epoch)
            writer.add_scalar("val/w-f1", val_f1_weighted, epoch)

            # storing best val metrics
            # TODO - add in more metrics
            if epoch != 0:
                if BEST_TEST['val_wt_f1_score'] < val_f1_weighted:
                    BEST_TEST['val_wt_f1_score'] = val_f1_weighted
                if BEST_TEST['val_accuracy'] < val_acc:
                    BEST_TEST['val_accuracy'] = val_acc

            logging.info("val - epoch ({}): | acc: {:.3f} | w f1: {:.3f} | micro f1: {:.3f} | macro f1: {:.3f}\n".format(
                epoch, val_acc, val_f1_weighted, val_f1_micro, val_f1_macro)
            )

            # checking early stopping per epoch
            patience -= 1
            adjust = False
            if lowest_f1_loss is None or valid_loss < lowest_f1_loss:
                adjust = True
                if lowest_f1_loss != None:
                    logging.info("val loss decreased {:.5f} -> {:.5f}! resetting patience to: {}".format(
                        lowest_f1_loss, valid_loss, reset_patience))

                # save the best model ",

                tdate = time.strftime('%Y%m%d')
                # TODO(dlee): add in support for balanced dataset.
                model_file_path = "/".join([
                    "/app/timeseries/multiclass_src/models/{dataset}",
                        '{tdate}-best_model-{experiment}-{seed}.pth'.format(
                            dataset, tdate, experiment, seed)])
                best_model_in_mem = io.BytesIO()
                torch.save(model, best_model_in_mem)
                logging.info("best model (in mem) is {}".format(model_file_path))

                patience = reset_patience
                lowest_f1_loss = valid_loss
                BEST_TEST['model_file_path'] = model_file_path

                # checking performance on test set
                # validation loop
                model.eval()
                test_losses = []
                test_preds, test_labels = np.array([]), np.array([])
                with torch.no_grad():
                    for i, (inputs, labels) in enumerate(test_loader):
                        labels_list = labels.cpu().numpy()
                        for label in labels_list:
                            test_count[int(label)] += 1

                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        output = model(inputs)
                        _, predicted = torch.max(output, 1)

                        pred_arr = predicted.cpu().numpy()
                        label_arr = labels.cpu().numpy()

                        test_labels = np.concatenate([test_labels, label_arr])
                        test_preds = np.concatenate([test_preds, pred_arr])

                        labels = labels.type(torch.int64)
                        if criterion_has_cm:
                            trans_labels = torch.zeros(len(labels), len(output[0])).to(device).scatter_(
                                1, labels.unsqueeze(1), 1.).to(device)
                            output = output.to(device)
                            batch_test_loss = criterion(
                                y_labels=trans_labels, y_preds=output)
                        else:
                            batch_test_loss = criterion(output, labels)

                    # adding in test loss
                    test_losses.append(batch_test_loss.detach().cpu().numpy())
                    test_acc = accuracy_score(y_true=test_labels, y_pred=test_preds)
                    test_f1_micro = f1_score(y_true=test_labels, y_pred=test_preds, average='micro')
                    test_f1_macro = f1_score(y_true=test_labels, y_pred=test_preds, average='macro')
                    test_f1_weighted = f1_score(y_true=test_labels, y_pred=test_preds, average='weighted')
                    test_class_f1s = f1_score(y_true=test_labels, y_pred=test_preds, average=None)
                    test_class_prs = precision_score(y_true=test_labels, y_pred=test_preds, average=None)
                    test_class_rec = recall_score(y_true=test_labels, y_pred=test_preds, average=None)

                    test_loss = np.mean(test_losses)
                    if experiment:
                        writer.add_scalar("test/test-loss", test_loss, epoch)
                        writer.add_scalar("test/accuracy", test_acc, epoch)
                        writer.add_scalar("test/micro-f1", test_f1_micro, epoch)
                        writer.add_scalar("test/macro-f1", test_f1_macro, epoch)
                        writer.add_scalar("test/w-f1", test_f1_weighted, epoch)
                        # adding per-class f1, precision, and recall
                        for i in range(N_CLASSES):
                            title = "test/class-" + str(i) + "-f1"
                            writer.add_scalar(title, np.array(test_class_f1s[i]).mean(), epoch)
                            title = "test/class-" + str(i) + "-precision"
                            writer.add_scalar(title, np.array(test_class_prs[i]).mean(), epoch)
                            title = "test/class-" + str(i) + "-recall"
                            writer.add_scalar(title, np.array(test_class_rec[i]).mean(), epoch)

                    if epoch != 0:
                        if BEST_TEST['loss'] > m_loss:
                            BEST_TEST['loss'] = m_loss
                            BEST_TEST['best-epoch'] = epoch
                        if BEST_TEST['test_wt_f1_score'] < test_f1_weighted:
                            BEST_TEST['test_wt_f1_score'] = test_f1_weighted
                        if BEST_TEST['test_accuracy'] < test_acc:
                            BEST_TEST['test_accuracy'] = test_acc

                    logging.info("test - epoch ({}): | loss: {:.4f} | acc: {:.3f} | w f1: {:.3f} | micro f1: {:.3f} | macro f1: {:.3f}".format(
                        epoch, test_loss, test_acc, test_f1_weighted, test_f1_micro, test_f1_macro))

            # if early stopping has begun, print it like this.
            if not adjust:
                logging.info("early stopping {}/{}...".format(reset_patience - patience, reset_patience))
            if patience <= 0:
                early_stopping = True
                logging.info("stopping!")
        
        
    # ----- FINAL EVALUATION STEP, USING FULLY TRAINED MODEL -----
    logging.info("--- finished training - entering final evaluation step\n")
    # TODO - why was i saving out the overfit model?
    # persist the best-in-memory model
    with open(model_file_path, 'wb') as f:
        f.write(best_model_in_mem.getbuffer())
    logging.info("{}: {}".format(BEST_TEST, model_file_path))
    # ----- recording results in a json.
    if torch.is_tensor(BEST_TEST['loss']):
        BEST_TEST['loss'] = BEST_TEST['loss'].item()
    if torch.is_tensor(BEST_TEST['test_wt_f1_score']):
        BEST_TEST['test_wt_f1_score'] = BEST_TEST['test_wt_f1_score'].item()
    if torch.is_tensor(BEST_TEST['val_wt_f1_score']):
        BEST_TEST['val_wt_f1_score'] = BEST_TEST['val_wt_f1_score'].item()
    BEST_TEST['loss'] = round(BEST_TEST['loss'], 5)
    BEST_TEST['test_wt_f1_score'] = round(BEST_TEST['test_wt_f1_score'], 5)
    BEST_TEST['val_wt_f1_score'] = round(BEST_TEST['val_wt_f1_score'], 5)
    BEST_TEST['train_count'] = train_count
    BEST_TEST['test_count'] = test_count
    BEST_TEST['val_count'] = valid_count
    pd.DataFrame({k: [v] for k, v in BEST_TEST.items()}).to_csv("{experiment_folder}.csv".format(
        experiment_folder=os.path.join("/app/timeseries/multiclass_src/results/{dataset}".format(dataset), experiment)))
    return


@click.command()
@click.option("--loss", required=True, default="ce")
@click.option("--dataset", required=True, default="iris")
@click.option("--epochs", required=True, default=100, type=int)
@click.option("--batch_size", required=True, default=32, type=int)
@click.option("--experiment", required=False, default="test")
@click.option("--gpu", required=False, default="0", type=str)
@click.option("--patience", required=True, type=int)
@click.option("--output", required=True, type=click.Path(exists=False))
@click.option("--init_weights", required=False, type=click.Path(exists=True))
def run(loss, dataset, epochs, batch_size, experiment, gpu, patience, output, init_weights):
    logging.info("Running with loss: {}, dataset: {}, epochs: {}, batch_size: {}, experiment: {}, gpu: {}, patience: {}, output: {}".format(
        loss, dataset, epochs, batch_size, experiment, gpu, patience, output, init_weights))

    batch_size = int(batch_size)
    epochs = int(epochs)
    train_for_dataset(
        dataset=dataset, 
        loss_metric=loss, 
        epochs=epochs, 
        seed=1, 
        experiment=experiment,
        cuda=gpu, 
        batch_size=batch_size, 
        patience=patience, 
        output_path=output, 
        init_weights=init_weights
    )


def main():
    os.environ['LC_ALL'] = 'C.UTF-8'
    os.environ['LANG'] = "C.UTF-8"
    run()


if __name__ == '__main__':
    main()

'''
example usage: 
python3 main.py --loss ce --dataset iris --epochs 100 --batch_size 32 --experiment test --gpu 0 --patience 10 --output /app/timeseries/multiclass_src/results/
python3 main.py --loss approx-f1 --dataset iris --epochs 10 --batch_size 32 --experiment test --gpu 0 --patience 10 --output /app/timeseries/multiclass_src/results/
'''
