import time
import torch
import glob
import numpy as np
import os
import segmentation_models_pytorch as smp
# flake8: noqa: E501


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py    # noqa: E501
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def to_device(data, device):
    if device.startswith('npu'):
        return {k: v.to(device, non_blocking=True) for k, v in data.items()}
    else:
        Warning("NPU Device is not supported")
    return data


def print_epoch_stats(phase,
                      epoch,
                      loss_meter,
                      batch_time,
                      data_time,
                      start_time,
                      iou=None):
    if iou:
        print(
            f"Epoch: [{epoch}] {phase} - TotalT: {(time.time() - start_time) / 60:.1f} min, "
            f"Loss: {loss_meter.avg:.4f}, Global IoU: {iou:.4f}")
    else:
        print(
            f"Epoch: [{epoch}] {phase} - TotalT: {(time.time() - start_time) / 60:.1f} min, "
            f"BatchT: {batch_time.avg:.3f}s, DataT: {data_time.avg:.3f}s, Loss: {loss_meter.avg:.4f}"
        )


def save_model(epoch, iou, model, optimizer, experiment_name, log_path,
               best_metric, best_metric_epoch):
    save_dict = {
        "model_name": experiment_name,
        "epoch_num": epoch,
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "iou": iou
    }
    old_model_path = glob.glob(f"{log_path}/best_iou*")
    if old_model_path:
        os.remove(old_model_path[0])
    torch.save(save_dict, f"{log_path}/best_iou_{epoch}_{iou:.4f}.pt")

    if iou > best_metric:
        best_metric, best_metric_epoch = iou, epoch

    return best_metric, best_metric_epoch


def create_data_loader(dataset, batch_size, shuffle, num_workers, pin_memory):
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers,
                                       pin_memory=pin_memory)


def generate_paths(base_path, pattern, replacements):
    paths = sorted(glob.glob(f"{base_path}/{pattern}"))
    return [
        list(map(lambda r: path.replace(*r), replacements)) for path in paths
    ]


def split_data(paths, train_perc, val_perc):
    np.random.seed(17)
    np.random.shuffle(paths)
    n = len(paths)
    return {
        "train": paths[:int(train_perc * n)],
        "val": paths[int(train_perc * n):int((train_perc + val_perc) * n)],
        "test": paths[int((train_perc + val_perc) * n):]
    }


def download_smp_model(model_params):
    model = getattr(smp, model_params["model_name"])(
        encoder_name=model_params["encoder_name"],
        encoder_weights=model_params["encoder_weights"],
        in_channels=model_params["in_channels"],
        classes=model_params["classes"])
    print(
        f"Downloaded and cached {model_params['model_name']} model with {model_params['encoder_name']} encoder."
    )
