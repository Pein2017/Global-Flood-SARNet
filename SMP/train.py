from loss_metrics import tp_fp_fn
from utils import AverageMeter, print_epoch_stats, to_device, save_model
import time
import torch

# flake8: noqa: E501
def train_epochs(model,
                 train_loader,
                 val_loader,
                 optimizer,
                 loss_func,
                 experiment_name,
                 log_path,
                 scheduler,
                 N_EPOCHS,
                 EARLY_STOP_THRESHOLD,
                 EARLY_STOP_PATIENCE,
                 DEVICE,
                 EPS,
                 start_epoch=1,
                 best_metric=0,
                 best_metric_epoch=0):
    
    early_stop_counter, best_val_early_stopping_metric = 0, 0
    start_time = time.time() # 全局的开始时间
    best_model_state = None
    for curr_epoch_num in range(start_epoch, N_EPOCHS + 1):
        epoch_start_time = time.time()  # 记录当前 Epoch 的开始时间

        # 训练阶段
        model.train()
        train_loss, batch_time, data_time = AverageMeter(), AverageMeter(), AverageMeter()
        end = time.time()

        for data in train_loader:
            data_time.update(time.time() - end)
            data = to_device(data, DEVICE)
            optimizer.zero_grad()
            preds = model(data["image"])
            loss = loss_func(preds, data["mask"].long())
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), data["image"].size(0))
            batch_time.update(time.time() - end)
            end = time.time()

        print_epoch_stats('Train', curr_epoch_num, train_loss, batch_time, data_time, epoch_start_time)

        # 验证阶段
        model.eval()
        val_loss, tps, fps, fns = AverageMeter(), 0, 0, 0
        with torch.no_grad():
            for data in val_loader:
                data = to_device(data, DEVICE)
                preds = model(data["image"])
                loss = loss_func(preds, data["mask"].long())
                val_loss.update(loss.item(), data["image"].size(0))
                preds_binary = (torch.softmax(preds, dim=1)[:, 1] > 0.5).long()
                tp, fp, fn = tp_fp_fn(preds_binary, data["mask"])
                tps += tp
                fps += fp
                fns += fn

        iou_global = tps / (tps + fps + fns + EPS)
        print_epoch_stats('Val', curr_epoch_num, val_loss, batch_time, data_time, epoch_start_time, iou_global)

        
        
        # 保存Scheduler和Model
        if curr_epoch_num > EARLY_STOP_THRESHOLD:
            scheduler.step(iou_global)
        if iou_global > best_metric:
            best_metric, best_metric_epoch = iou_global, curr_epoch_num
            best_model_state = {
                "model_name": experiment_name,
                "epoch_num": curr_epoch_num,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "iou": iou_global
            }
        if iou_global < best_val_early_stopping_metric:
            early_stop_counter += 1
        else:
            best_val_early_stopping_metric, early_stop_counter = iou_global, 0
        if early_stop_counter > EARLY_STOP_PATIENCE:
            print("Early Stopping")
            break

    # 训练结束后保存最佳模型
    if best_model_state:
        save_model(best_model_state, log_path)

    print(f"Best IOU: {best_metric} at Epoch: {best_metric_epoch}")

    # 返回有用的信息
    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "best_metric": best_metric,
        "best_metric_epoch": best_metric_epoch,
        "total_epochs": curr_epoch_num,
        "training_time": time.time() - start_time
    }
