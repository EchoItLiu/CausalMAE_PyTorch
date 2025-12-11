import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils
from utils import maxmin1, accuracy_fn, calculate_performance_metrics, plot_cm
import numpy as np
#
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, classification_report




def train_class_batch(model, samples, mask, is_pt, target, criterion):
    outputs = model(samples, mask, is_pt) # 
    
    loss = criterion(outputs, target) 
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    print ('***op1:', optimizer)
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    total_loss = 0
    all_preds = []
    all_labels = []
    batch_metrics = []
    correct = 0
    total = 0
    
    mask, is_pt = None, False

    
    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()


    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            # optim_factory  get_parameter_groups
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        
        if loss_scaler is None:
            print ('fp16-2')            
        # fp16
        # model = model.half()
            samples = samples.half()
        # 
        # with torch.cuda.amp.autocast(enabled=False):
            loss, output = train_class_batch(
                model, samples, mask, is_pt, targets, criterion)
        
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, samples, mask, is_pt, targets, criterion)

        # scalar
        loss_value = loss.item()

        

        _, preds = torch.max(output, 1)
  
        
        

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())


        batch_y = targets.cpu().numpy()
        batch_pred = preds.cpu().numpy()


        

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        
        if loss_scaler is None:
            loss /= update_freq # update_freq = 1
            model.backward(loss)
            model.step()
            # 
            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)

        
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                # 
                if model_ema is not None:
                    model_ema.update(model) # 
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()
        

        
        try:
            batch_precision = precision_score(batch_y, batch_pred, pos_label=0)
            batch_recall = recall_score(batch_y, batch_pred, pos_label=0)
            batch_f1 = f1_score(batch_y, batch_pred, pos_label=0)
            batch_acc = accuracy_score(batch_y, batch_pred)
            correct += (batch_pred == batch_y).sum().item()
            total += samples.size(0)
            
            tn = ((1 - batch_y) & (1 - batch_pred)).sum()
            fp = ((1 - batch_y) & batch_pred).sum()
            batch_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            metric_logger.update(b_prec = batch_precision)
            metric_logger.update(b_acc = batch_acc)
            metric_logger.update(b_recall = batch_recall)
            metric_logger.update(b_f1 = batch_f1)
            metric_logger.update(b_sp = batch_specificity)
            metric_logger.update(loss = loss_value)



            # Logger Print
            metric_logger.update(loss_scale = loss_scale_value)
            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])
            metric_logger.update(lr=max_lr)
            metric_logger.update(min_lr=min_lr)
            
            print ()
            weight_decay_value = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]
            metric_logger.update(weight_decay=weight_decay_value)
            metric_logger.update(grad_norm=grad_norm)

        
            batch_metrics.append([batch_acc, batch_precision, batch_recall, batch_f1, batch_specificity])

        except:
            pass

        total_loss += loss.item() * samples.size(0)

    avg_loss = total_loss / len(data_loader.dataset)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels) 
    all_labels = all_labels.astype(np.int32)
    all_preds = all_preds.astype(np.int32)    

    accuracy =  correct / total
    precision = precision_score(all_labels, all_preds, pos_label=0)
    recall = recall_score(all_labels, all_preds, pos_label=0)
    f1 = f1_score(all_labels, all_preds, pos_label=0)





    tn = ((1 - all_labels) & (1 - all_preds)).sum()
    fp = ((1 - all_labels) & all_preds).sum()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    if len(batch_metrics) > 0:
        batch_metrics = np.array(batch_metrics)
        acc_std = np.std(batch_metrics[:, 0])
        precision_std = np.std(batch_metrics[:, 1])
        recall_std = np.std(batch_metrics[:, 2])
        f1_std = np.std(batch_metrics[:, 3])
        specificity_std = np.std(batch_metrics[:, 4])
    else:
        acc_std = precision_std = recall_std = f1_std = specificity_std = 0.0

    report = classification_report(all_labels, all_preds, target_names=['Pathological', 'Healthy'])

    print ('**********report**********:', report)

    epoch_indicators = {
                'avg_loss': avg_loss,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'specificity': specificity,
                'acc_std': acc_std,
                'precision_std': precision_std,
                'recall_std': recall_std,
                'f1_std': f1_std,
                'specificity_std': specificity_std,
                'preds': all_preds,
                'labels': all_labels
            }
    

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, epoch_indicators







@torch.no_grad()
def evaluate(data_loader, model, device, args, log_writer):

    total_loss_eva = 0
    all_preds_eva = []
    all_labels_eva = []
    batch_metrics_eva = []
    correct_eva = 0
    total_eva = 0    

    criterion = nn.CrossEntropyLoss() 
    
    cm_t = np.zeros((2,2))
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()

    
    for batch in metric_logger.log_every(data_loader, 10, header):
        grfs = batch[0]
        targets = batch[-1]
        
        grfs = grfs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        mask, is_pt = None, False
        #
        #

        with torch.cuda.amp.autocast():
            outputs = model(grfs, mask, is_pt) # 
            loss = criterion(outputs, targets)


        loss_value = loss.item()
        batch_size = grfs.shape[0]
        
        _, preds = torch.max(outputs, 1)   
        
        
        all_preds_eva.extend(preds.cpu().numpy())
        all_labels_eva.extend(targets.cpu().numpy())


        batch_y_eva = targets.cpu().numpy()
        batch_pred_eva = preds.cpu().numpy()        
        

        batch_precision_eva = precision_score(batch_y_eva, batch_pred_eva, pos_label=0)
        batch_recall_eva = recall_score(batch_y_eva, batch_pred_eva, pos_label=0)
        batch_f1_eva = f1_score(batch_y_eva, batch_pred_eva, pos_label=0)
        batch_acc_eva = accuracy_score(batch_y_eva, batch_pred_eva)
        correct_eva += (batch_pred_eva == batch_y_eva).sum().item()
        total_eva += grfs.size(0) 
        
        tn = ((1 - batch_y_eva) & (1 - batch_pred_eva)).sum()
        fp = ((1 - batch_y_eva) & batch_pred_eva).sum()
        batch_specificity_eva = tn / (tn + fp) if (tn + fp) > 0 else 0
        #

        
        metric_logger.update(b_prec_eva = batch_precision_eva)
        metric_logger.update(b_acc_eva = batch_acc_eva)
        metric_logger.update(b_recall_eva = batch_recall_eva)
        metric_logger.update(b_f1_eva = batch_f1_eva)
        metric_logger.update(b_sp_eva = batch_specificity_eva)
        metric_logger.update(loss_eva = loss_value)

        # Tensorboard
        if log_writer is not None:
            # 
            log_writer.update(batch_precision_eva = batch_precision_eva, head="eva_precision")
            log_writer.update(batch_acc_eva = batch_acc_eva, head="eva_acc")
            log_writer.update(b_recall_eva = batch_recall_eva, head="eva_recall")
            log_writer.update(b_f1_eva = batch_f1_eva, head="eva_f1-score")
            log_writer.update(batch_specificity_eva = batch_specificity_eva, head="eva_sp")
            log_writer.update(loss_eva = loss_value, head="loss_eva")
    
        #      
     
        batch_metrics_eva.append([batch_acc_eva, batch_precision_eva, batch_recall_eva, batch_f1_eva, batch_specificity_eva])
        total_loss_eva += loss_value * grfs.size(0)
    


    # 
    avg_loss_eva = total_loss_eva / len(data_loader.dataset)
    all_preds_eva = np.array(all_preds_eva)
    all_labels_eva = np.array(all_labels_eva) 

    all_labels_eva = all_labels_eva.astype(np.int32)
    all_preds_eva = all_preds_eva.astype(np.int32)   
    


    accuracy_eva =  correct_eva / total_eva   
    precision_eva = precision_score(all_labels_eva, all_preds_eva, pos_label=0)
    recall_eva = recall_score(all_labels_eva, all_preds_eva, pos_label=0)
    f1_eva = f1_score(all_labels_eva, all_preds_eva, pos_label=0)

 

    tn = ((1 - all_labels_eva) & (1 - all_preds_eva)).sum()
    fp = ((1 - all_labels_eva) & all_preds_eva).sum()
    specificity_eva = tn / (tn + fp) if (tn + fp) > 0 else 0


    
    if len(batch_metrics_eva) > 0:
        batch_metrics_eva = np.array(batch_metrics_eva)
        acc_std_eva = np.std(batch_metrics_eva[:, 0])
        precision_std_eva = np.std(batch_metrics_eva[:, 1])
        recall_std_eva = np.std(batch_metrics_eva[:, 2])
        f1_std_eva = np.std(batch_metrics_eva[:, 3])
        specificity_std_eva = np.std(batch_metrics_eva[:, 4])
    else:
        acc_std_eva = precision_std_eva = recall_std_eva = f1_std_eva = specificity_std_eva = 0.0


    report_eva = classification_report(all_labels_eva, all_preds_eva, target_names=['Pathological', 'Healthy'])

    print ('**********report_eva**********:', report_eva)

    table_indicators = {
                'avg_loss_eva': avg_loss_eva,
                'accuracy_eva': accuracy_eva,
                'precision_eva': precision_eva,
                'recall_eva': recall_eva,
                'f1_eva': f1_eva,
                'specificity_eva': specificity_eva,
                'acc_std_eva': acc_std_eva,
                'precision_std_eva': precision_std_eva,
                'recall_std_eva': recall_std_eva,
                'f1_std_eva': f1_std_eva,
                'specificity_std_eva': specificity_std_eva,
                'preds_eva': all_preds_eva,
                'labels_eva': all_labels_eva
            }
    

    metric_logger.synchronize_between_processes()
 

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, table_indicators
